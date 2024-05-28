/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <rasterize_points.h>
#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <rasterizer.h>
#include <fstream> 
#include <imgui_internal.h>



 // Define the types and sizes that make up the contents of each Gaussian 
 // in the trained model.

struct AnchorPoint
{
	float pos[3];
	float normal[3];
	float offset[30];
	float feat[32];
	float opacity;
	float scale[6];
	float rot[4];
};


float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

float inverse_sigmoid(const float m1)
{
	return log(m1 / (1.0f - m1));
}

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif

// Load the Gaussians from the given file.
int loadPly(const char* filename,
	std::vector<float>& pos,
	std::vector<float>& normal,
	std::vector<float>& offset,
	std::vector<float>& feat,
	std::vector<float>& opacity,
	std::vector<float>& scale1,
	std::vector<float>& scale2,
	std::vector<float>& rot,
	std::vector<float>& gs_pos,
	sibr::Vector3f& minn,
	sibr::Vector3f& maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		SIBR_ERR << "Unable to find model's PLY file, attempted:\n" << filename << std::endl;

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Anchors points contained
	SIBR_LOG << "Loading " << count << " Anchor points" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<AnchorPoint> points(count);
	infile.read((char*)points.data(), count * sizeof(AnchorPoint));

	// Resize our SoA data
	pos.resize(count * 3);
	normal.resize(count * 3);
	offset.resize(count * 30);
	feat.resize(count * 32);
	opacity.resize(count);
	scale1.resize(count * 3);
	scale2.resize(count * 3);
	rot.resize(count * 4);
	gs_pos.resize(count * 30);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	minn = sibr::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	maxx = -minn;
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f pos = sibr::Vector3f(points[i].pos[0], points[i].pos[1], points[i].pos[2]);
		maxx = maxx.cwiseMax(pos);
		minn = minn.cwiseMin(pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		sibr::Vector3f pos = sibr::Vector3f(points[i].pos[0], points[i].pos[1], points[i].pos[2]);
		sibr::Vector3f rel = (pos - minn).array() / (maxx - minn).array();
		sibr::Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		sibr::Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
		};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	for (int k = 0; k < count; k++)
	{
		int i = k;//mapp[k].second;
		for (int j = 0; j < 3; j++)
		{
			pos[3 * k + j] = points[i].pos[j];
		}
		for (int j = 0; j < 3; j++)
		{
			normal[3 * k + j] = points[i].normal[j];
		}
		for (int j = 0; j < 32; j++)
		{
			feat[32 * k + j] = points[i].feat[j];
		}
		opacity[k] = points[i].opacity;
		for (int j = 0; j < 3; j++)
		{
			scale1[3 * k + j] = exp(points[i].scale[j]);
			scale2[3 * k + j] = points[i].scale[j + 3];
		}
		for (int j = 0; j < 4; j++)
		{
			rot[4 * k + j] = points[i].rot[j];
		}
		for (int j = 0; j < 30; j++)
		{
			offset[30 * k + j] = points[i].offset[(j % 3) * 10 + (j / 3)];
			gs_pos[30 * k + j] = pos[3 * k + (j % 3)] + scale1[3 * k + (j % 3)] * offset[30 * k + j];
		}
	}
	return count;
}

void savePly(const char* filename,
	std::vector<float>& pos,
	std::vector<float>& normal,
	std::vector<float>& offset,
	std::vector<float>& feat,
	std::vector<float>& opacity,
	std::vector<float>& scale1,
	std::vector<float>& scale2,
	std::vector<float>& rot,
	const sibr::Vector3f& minn,
	const sibr::Vector3f& maxx)
{
	// Read all Gaussians at once (AoS)
	int count = 0;
	for (int i = 0; i < pos.size() / 3; i++)
	{
		if (pos[3 * i] < minn.x() || pos[3 * i + 1] < minn.y() || pos[3 * i + 2] < minn.z() ||
			pos[3 * i] > maxx.x() || pos[3 * i + 1] > maxx.y() || pos[3 * i + 2] > maxx.z())
			continue;
		count++;
	}
	std::vector<AnchorPoint> points(count);

	// Output number of Anchor points contained
	SIBR_LOG << "Saving " << count << " Anchor points" << std::endl;

	std::ofstream outfile(filename, std::ios_base::binary);

	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

	std::string prop_pos[] = { "x", "y", "z" };
	std::string prop_normal[] = { "nx", "ny", "nz" };
	std::string props_offset[] = {
		"f_offset_0", "f_offset_1", "f_offset_2", "f_offset_3", "f_offset_4", "f_offset_5",
		"f_offset_6", "f_offset_7", "f_offset_8", "f_offset_9", "f_offset_10", "f_offset_11",
		"f_offset_12", "f_offset_13", "f_offset_14", "f_offset_15", "f_offset_16", "f_offset_17",
		"f_offset_18", "f_offset_19", "f_offset_20", "f_offset_21", "f_offset_22", "f_offset_23",
		"f_offset_24", "f_offset_25", "f_offset_26", "f_offset_27", "f_offset_28", "f_offset_29"
	};
	std::string props_feat[] = {
		"f_anchor_feat_0", "f_anchor_feat_1", "f_anchor_feat_2", "f_anchor_feat_3", "f_anchor_feat_4",
		"f_anchor_feat_5", "f_anchor_feat_6", "f_anchor_feat_7", "f_anchor_feat_8", "f_anchor_feat_9",
		"f_anchor_feat_10", "f_anchor_feat_11", "f_anchor_feat_12", "f_anchor_feat_13", "f_anchor_feat_14",
		"f_anchor_feat_15", "f_anchor_feat_16", "f_anchor_feat_17", "f_anchor_feat_18", "f_anchor_feat_19",
		"f_anchor_feat_20", "f_anchor_feat_21", "f_anchor_feat_22", "f_anchor_feat_23", "f_anchor_feat_24",
		"f_anchor_feat_25", "f_anchor_feat_26", "f_anchor_feat_27", "f_anchor_feat_28", "f_anchor_feat_29",
		"f_anchor_feat_30", "f_anchor_feat_31"
	};
	std::string props_opacity[] = { "opacity" };
	std::string props_base_offset[] = { "scale_0", "scale_1", "scale_2" };
	std::string props_base_scale[] = { "scale_3", "scale_4", "scale_5" };
	std::string props_rot[] = { "rot_0", "rot_1", "rot_2", "rot_3" };

	for (auto s : prop_pos)
		outfile << "property float " << s << std::endl;
	for (auto s : prop_normal)
		outfile << "property float " << s << std::endl;
	for (auto s : props_offset)
		outfile << "property float " << s << std::endl;
	for (auto s : props_feat)
		outfile << "property float " << s << std::endl;
	for (auto s : props_opacity)
		outfile << "property float " << s << std::endl;
	for (auto s : props_base_offset)
		outfile << "property float " << s << std::endl;
	for (auto s : props_base_scale)
		outfile << "property float " << s << std::endl;

	count = 0;
	for (int i = 0; i < pos.size() / 3; i++)
	{
		if (pos[3 * i] < minn.x() || pos[3 * i + 1] < minn.y() || pos[3 * i + 2] < minn.z() ||
			pos[3 * i] > maxx.x() || pos[3 * i + 1] > maxx.y() || pos[3 * i + 2] > maxx.z())
			continue;
		for (int j = 0; j < 3; j++)
		{
			points[count].pos[j] = pos[3 * i + j];
		}
		for (int j = 0; j < 3; j++)
		{
			points[i].normal[j] = normal[3 * i + j];
		}
		for (int j = 0; j < 30; j++)
		{
			points[i].offset[(j % 3) * 10 + (j / 3)] = offset[30 * i + j];
		}
		for (int j = 0; j < 32; j++)
		{
			points[i].feat[j] = feat[32 * i + j];
		}
		points[i].opacity = opacity[i];
		for (int j = 0; j < 3; j++)
		{
			points[i].scale[j] = scale1[3 * i + j];
			points[i].scale[j + 3] = scale2[3 * i + j];
		}
		for (int j = 0; j < 4; j++)
		{
			points[i].rot[j] = rot[4 * i + j];
		}
		count++;
	}
	outfile.write((char*)points.data(), sizeof(AnchorPoint) * points.size());
}

namespace sibr
{
	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target. 
	class BufferCopyRenderer
	{

	public:

		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }
		int& width() { return _width.get(); }
		int& height() { return _height.get(); }

	private:

		GLShader			_shader;
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};
}

bool isFileExists_fopen(std::string& name) {
	if (FILE* file = fopen(name.c_str(), "r")) {
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
		};
	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, std::string plyPath, bool* messageRead, bool white_bg, bool useInterop, int device, int appearance_id, bool add_opacity_dist, bool add_cov_dist, bool add_color_dist) :
	_scene(ibrScene),
	_dontshow(messageRead),
	_appearance_id(appearance_id),
	_add_opacity_dist(add_opacity_dist),
	_add_cov_dist(add_cov_dist),
	_add_color_dist(add_color_dist),
	_libtorch_device(torch::kCUDA),
	sibr::ViewBase(render_w, render_h)
{
	int num_devices;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices)
	{
		if (num_devices == 0)
			SIBR_ERR << "No CUDA devices detected!";
		else
			SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));
	if (prop.major < 7)
	{
		SIBR_ERR << "Sorry, need at least compute capability 7.0+!";
	}

	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = render_w;
	_copyRenderer->height() = render_h;

	std::vector<uint> imgs_ulr;
	const auto& cams = ibrScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

	// 获取file的文件夹名， file是const char*
	std::string ply_path = plyPath + "point_cloud.ply";
	std::string opacity_mlp_path = plyPath + "opacity_mlp.pt";
	std::string cov_mlp_path = plyPath + "cov_mlp.pt";
	std::string color_mlp_path = plyPath + "color_mlp.pt";
	std::string appearance_path = plyPath + "embedding_appearance.pt";
	SIBR_LOG << "Loading models from: " << plyPath << std::endl;
	SIBR_LOG << "opacity_mlp : " << isFileExists_fopen(opacity_mlp_path) << std::endl;
	SIBR_LOG << "cov_mlp : " << isFileExists_fopen(cov_mlp_path) << std::endl;
	SIBR_LOG << "color_mlp : " << isFileExists_fopen(color_mlp_path) << std::endl;
	SIBR_LOG << "embedding_appearance : " << isFileExists_fopen(appearance_path) << std::endl;
	opacity_mlp_module = torch::jit::load(opacity_mlp_path, _libtorch_device);
	color_mlp_module = torch::jit::load(color_mlp_path, _libtorch_device);
	cov_mlp_module = torch::jit::load(cov_mlp_path, _libtorch_device);
	if (isFileExists_fopen(appearance_path))
	{
		appearance_module = torch::jit::load(appearance_path, _libtorch_device);
		SIBR_LOG << "appearance code id : " << _appearance_id << std::endl;
		_add_appearance = true;
	}
	else
	{
		_add_appearance = false;
	}

	// Load the PLY data (AoS) to the GPU (SoA)
	count = loadPly(ply_path.c_str(), anchor_pos, anchor_normal, anchor_offset, anchor_feature, anchor_opacity, anchor_scale_1, anchor_scale_2, anchor_rotation, gaussian_pos, _scenemax, _scenemin);
	ak_pos_all = torch::from_blob(anchor_pos.data(), { count , 3 }, torch::kFloat32).to(_libtorch_device);
	gs_pos_all = torch::from_blob(gaussian_pos.data(), { count, 30 }, torch::kFloat32).to(_libtorch_device);
	ak_feat_all = torch::from_blob(anchor_feature.data(), { count, 32 }, torch::kFloat32).to(_libtorch_device);
	ak_rot_all = torch::from_blob(anchor_rotation.data(), { count, 4 }, torch::kFloat32).to(_libtorch_device);
	ak_rot_all = torch::nn::functional::normalize(ak_rot_all, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
	ak_scale_1 = torch::from_blob(anchor_scale_1.data(), { count, 3 }, torch::kFloat32).to(_libtorch_device);
	ak_scale_2 = torch::from_blob(anchor_scale_2.data(), { count, 3 }, torch::kFloat32).to(_libtorch_device);
	ak_scale_2 = torch::exp(ak_scale_2);

	_boxmin = _scenemin;
	_boxmax = _scenemax;

	// Create space for view parameters
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&rect_cuda, 2 * count * sizeof(int)));

	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	_gaussianRenderer = new GaussianSurfaceRenderer();

	// Create GL buffer ready for CUDA/GL interop
	glCreateBuffers(1, &imageBuffer);
	glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

	if (useInterop)
	{
		if (cudaPeekAtLastError() != cudaSuccess)
		{
			SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
		}
		cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
		useInterop &= (cudaGetLastError() == cudaSuccess);
	}
	if (!useInterop)
	{
		fallback_bytes.resize(render_w * render_h * 3 * sizeof(float));
		cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
		_interop_failed = true;
	}

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);
	geomBufferFunc_filter = resizeFunctional(&geomPtr_filter, allocdGeom_filter);
	binningBufferFunc_filter = resizeFunctional(&binningPtr_filter, allocdBinning_filter);
	imgBufferFunc_filter = resizeFunctional(&imgPtr_filter, allocdImg_filter);
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr& newScene)
{
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto& cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye)
{
	if (currMode == "Initial Points")
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{
		// Convert view and projection to target coordinate system
		auto view_mat = eye.view();
		auto proj_mat = eye.viewproj();
		view_mat.row(1) *= -1;
		view_mat.row(2) *= -1;
		proj_mat.row(1) *= -1;

		float scale_modifier = 1.0f;
		torch::Tensor viewmatrix = torch::from_blob(view_mat.data(), { 4,4 }, torch::kFloat32).to(_libtorch_device);
		torch::Tensor projmatrix = torch::from_blob(proj_mat.data(), { 4,4 }, torch::kFloat32).to(_libtorch_device);
		// Compute additional view parameters
		float tan_fovy = tan(eye.fovy() * 0.5f);
		float tan_fovx = tan_fovy * eye.aspect();

		const int H = _resolution.y();
		const int W = _resolution.x();
		torch::Tensor radii = torch::full({ count }, 0, ak_pos_all.options().dtype(torch::kInt32));

		if (count != 0)
		{
			int M = 0;
			CudaRasterizer::Rasterizer::visible_filter(
				geomBufferFunc_filter,
				binningBufferFunc_filter,
				imgBufferFunc_filter,
				count, M,
				W, H,
				ak_pos_all.contiguous().data<float>(),
				ak_scale_1.contiguous().data_ptr<float>(),
				scale_modifier,
				ak_rot_all.contiguous().data_ptr<float>(),
				nullptr,
				viewmatrix.contiguous().data<float>(),
				projmatrix.contiguous().data<float>(),
				tan_fovx,
				tan_fovy,
				FALSE,
				radii.contiguous().data<int>()
			);
		}

		radii = (radii > 0).to(_libtorch_device);

		torch::Tensor gs_pos = gs_pos_all.index({ radii });
		torch::Tensor gs_scale = ak_scale_2.index({ radii });
		torch::Tensor ak_feat = ak_feat_all.index({ radii });
		torch::Tensor ak_pos = ak_pos_all.index({ radii });

		sibr::Vector3f eye_pos = eye.position();
		torch::Tensor eye_pos_tensor = torch::from_blob(eye_pos.data(), { 1,3 }, torch::kFloat32).to(_libtorch_device);
		torch::Tensor ob_view = ak_pos - eye_pos_tensor;
		torch::Tensor ob_dist = torch::sqrt(torch::sum(torch::pow(ob_view, 2), 1)).view({ -1, 1 }).to(_libtorch_device);
		ob_view = ob_view.div(ob_dist);

		torch::Tensor cat_local_view = torch::cat({ ak_feat, ob_view, ob_dist }, 1).to(_libtorch_device);
		torch::Tensor cat_local_view_wodist;
		if (!_add_opacity_dist || !_add_cov_dist || !_add_color_dist)
		{
			cat_local_view_wodist = torch::cat({ ak_feat, ob_view }, 1).to(_libtorch_device);
		}


		torch::NoGradGuard no_grad;
		torch::Tensor neural_opacity;
		if(_add_opacity_dist)
			neural_opacity = opacity_mlp_module.forward({ cat_local_view }).toTensor().to(_libtorch_device);
		else
			neural_opacity = opacity_mlp_module.forward({ cat_local_view_wodist }).toTensor().to(_libtorch_device);
		neural_opacity = neural_opacity.reshape({ -1, 1 });
		torch::Tensor mask = (neural_opacity > 0).view({ -1 }).to(_libtorch_device);

		int M = ak_pos.size(0);
		//SIBR_LOG << "ToTal " << P << " Gaussian points" << std::endl;

		torch::Tensor scale_rot;
		if(_add_cov_dist)
			scale_rot = cov_mlp_module.forward({ cat_local_view }).toTensor().to(_libtorch_device);
		else
			scale_rot = cov_mlp_module.forward({ cat_local_view_wodist }).toTensor().to(_libtorch_device);
		scale_rot = scale_rot.reshape({ M * 10, 7 });

		torch::Tensor gs_color;
		if (_add_appearance)
		{
			torch::Tensor camera_indices = torch::ones({ M }, torch::kLong).to(_libtorch_device);
 			camera_indices = camera_indices * _appearance_id;
			torch::Tensor appearance = appearance_module.forward({ camera_indices }).toTensor().to(_libtorch_device);
			if (_add_color_dist)
			{
				cat_local_view = torch::cat({ cat_local_view, appearance }, 1).to(_libtorch_device);
				gs_color = color_mlp_module.forward({ cat_local_view }).toTensor().to(_libtorch_device);
			}
			else
			{
				cat_local_view_wodist = torch::cat({ cat_local_view_wodist, appearance }, 1).to(_libtorch_device);
				gs_color = color_mlp_module.forward({ cat_local_view_wodist }).toTensor().to(_libtorch_device);
			}
		}
		else
		{
			if (_add_color_dist)
			{
				gs_color = color_mlp_module.forward({ cat_local_view }).toTensor().to(_libtorch_device);
			}
			else
			{
				gs_color = color_mlp_module.forward({ cat_local_view_wodist }).toTensor().to(_libtorch_device);
			}
		}
		gs_color = gs_color.reshape({ M * 10, 3 });

		gs_scale = gs_scale.unsqueeze(0).repeat({ 10, 1, 1 }).transpose(0, 1).reshape({ M * 10, 3 });
		gs_pos = gs_pos.reshape({ M * 10, 3 });
		torch::Tensor all = torch::cat({ gs_pos, gs_scale, gs_color, scale_rot, neural_opacity }, -1).to(_libtorch_device);;

		all = all.index({ mask });
		int P = all.size(0);

		gs_pos = all.narrow(-1, 0, 3);
		gs_scale = torch::sigmoid(all.narrow(-1, 9, 3)).mul(all.narrow(-1, 3, 3));
		gs_color = all.narrow(-1, 6, 3);
		torch::Tensor gs_rot = torch::nn::functional::normalize(all.narrow(-1, 12, 4), torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
		torch::Tensor gs_opacity = all.narrow(-1, 16, 1);

		// Copy frame-dependent data to GPU
		CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));

		// Map OpenGL buffer resource for use with CUDA
		size_t bytes;
		float* image_cuda = nullptr;
		if (!_interop_failed)
		{
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
			CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
		}
		else
		{
			image_cuda = fallbackBufferCuda;
		}

		// Rasterize
		int* rects = _fastCulling ? rect_cuda : nullptr;
		float* boxmin = _cropping ? (float*)&_boxmin : nullptr;
		float* boxmax = _cropping ? (float*)&_boxmax : nullptr;
		CudaRasterizer::Rasterizer::forward(
			geomBufferFunc,
			binningBufferFunc,
			imgBufferFunc,
			P, 1, 16,
			background_cuda,
			_resolution.x(), _resolution.y(),
			gs_pos.contiguous().data<float>(),
			nullptr,
			gs_color.contiguous().data<float>(),
			gs_opacity.contiguous().data<float>(),
			gs_scale.contiguous().data<float>(),
			_scalingModifier,
			gs_rot.contiguous().data<float>(),
			nullptr,
			view_cuda,
			proj_cuda,
			cam_pos_cuda,
			tan_fovx,
			tan_fovy,
			false,
			image_cuda,
			nullptr,
			rects,
			boxmin,
			boxmax
		);

		if (!_interop_failed)
		{
			// Unmap OpenGL resource for use with OpenGL
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
		}
		else
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
		}
		// Copy image contents to framebuffer
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());

	}

	if (cudaPeekAtLastError() != cudaSuccess)
	{
		SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
	}

}

void sibr::GaussianView::onUpdate(Input& input)
{
}

void sibr::GaussianView::onGUI()
{
	// Generate and update UI elements
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str()))
	{
		if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
		{
			if (ImGui::Selectable("Splats"))
				currMode = "Splats";
			if (ImGui::Selectable("Initial Points"))
				currMode = "Initial Points";
			//if (ImGui::Selectable("Ellipsoids"))
			//	currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")
	{
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);
	}
	ImGui::Checkbox("Fast culling", &_fastCulling);

	ImGui::Checkbox("Crop Box", &_cropping);
	if (_cropping)
	{
		ImGui::SliderFloat("Box Min X", &_boxmin.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Min Y", &_boxmin.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Min Z", &_boxmin.z(), _scenemin.z(), _scenemax.z());
		ImGui::SliderFloat("Box Max X", &_boxmax.x(), _scenemin.x(), _scenemax.x());
		ImGui::SliderFloat("Box Max Y", &_boxmax.y(), _scenemin.y(), _scenemax.y());
		ImGui::SliderFloat("Box Max Z", &_boxmax.z(), _scenemin.z(), _scenemax.z());
		ImGui::InputText("File", _buff, 512);
		if (ImGui::Button("Save"))
		{
			savePly(_buff,
				anchor_pos,
				anchor_normal,
				anchor_offset,
				anchor_feature,
				anchor_opacity,
				anchor_scale_1,
				anchor_scale_2,
				anchor_rotation,
				_boxmin,
				_boxmax
			);
		}
	}

	ImGui::End();

	if (!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");

	if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"\
			" It did NOT work for your current configuration.\n"\
			" For highest performance, OpenGL and CUDA must run on the same\n"\
			" GPU on an OS that supports interop.You can try to pass a\n"\
			" non-zero index via --device on a multi-GPU system, and/or try\n" \
			" attaching the monitors to the main CUDA card.\n"\
			" On a laptop with one integrated and one dedicated GPU, you can try\n"\
			" to set the preferred GPU via your operating system.\n\n"\
			" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

		ImGui::Separator();

		if (ImGui::Button("  OK  ")) {
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}
}

sibr::GaussianView::~GaussianView()
{
	// Cleanup
	//cudaFree(pos_cuda);
	//cudaFree(rot_cuda);
	//cudaFree(scale_cuda);
	//cudaFree(opacity_cuda);
	//cudaFree(color_cuda);

	cudaFree(view_cuda);
	cudaFree(proj_cuda);
	cudaFree(cam_pos_cuda);
	cudaFree(background_cuda);
	cudaFree(rect_cuda);

	if (!_interop_failed)
	{
		cudaGraphicsUnregisterResource(imageBufferCuda);
	}
	else
	{
		cudaFree(fallbackBufferCuda);
	}
	glDeleteBuffers(1, &imageBuffer);

	if (geomPtr)
		cudaFree(geomPtr);
	if (binningPtr)
		cudaFree(binningPtr);
	if (imgPtr)
		cudaFree(imgPtr);

	delete _copyRenderer;
}
