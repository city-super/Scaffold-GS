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
#pragma once


# include "Config.hpp"
# include <core/renderer/RenderMaskHolder.hpp>
# include <core/scene/BasicIBRScene.hpp>
# include <core/system/SimpleTimer.hpp>
# include <core/system/Config.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/view/ViewBase.hpp>
# include <core/renderer/CopyRenderer.hpp>
# include <core/renderer/PointBasedRenderer.hpp>
# include <memory>
# include <core/graphics/Texture.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <torch/script.h>
#include <functional>
# include "GaussianSurfaceRenderer.hpp"

namespace CudaRasterizer
{
	class Rasterizer;
}

namespace sibr {

	class BufferCopyRenderer;
	class BufferCopyRenderer2;

	/**
	 * \class RemotePointView
	 * \brief Wrap a ULR renderer with additional parameters and information.
	 */
	class SIBR_EXP_ULR_EXPORT GaussianView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(GaussianView);

	public:

		/**
		 * Constructor
		 * \param ibrScene The scene to use for rendering.
		 * \param render_w rendering width
		 * \param render_h rendering height
		 */
		GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, std::string plyPath, bool* message_read, bool white_bg = false, bool useInterop = true, int device = 0,
			int appearance_id = 0, bool add_opacity_dist = true, bool add_cov_dist = true, bool add_color_dist = true);

		/** Replace the current scene.
		 *\param newScene the new scene to render */
		void setScene(const sibr::BasicIBRScene::Ptr& newScene);

		/**
		 * Perform rendering. Called by the view manager or rendering mode.
		 * \param dst The destination rendertarget.
		 * \param eye The novel viewpoint.
		 */
		void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;

		/**
		 * Update inputs (do nothing).
		 * \param input The inputs state.
		 */
		void onUpdate(Input& input) override;

		/**
		 * Update the GUI.
		 */
		void onGUI() override;

		/** \return a reference to the scene */
		const std::shared_ptr<sibr::BasicIBRScene>& getScene() const { return _scene; }

		virtual ~GaussianView() override;

		bool* _dontshow;

	protected:

		std::string currMode = "Splats";

		bool _cropping = false;
		sibr::Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
		char _buff[512] = "cropped.ply";

		bool _fastCulling = true;
		int _device = 0;

		int count;
		int _appearance_id;
		bool _add_appearance;
		bool _add_opacity_dist;
		bool _add_cov_dist;
		bool _add_color_dist;
		torch::Device _libtorch_device;
		torch::jit::script::Module opacity_mlp_module;
		torch::jit::script::Module color_mlp_module;
		torch::jit::script::Module cov_mlp_module;
		torch::jit::script::Module appearance_module;

		std::vector<float> anchor_pos;
		std::vector<float> anchor_normal;
		std::vector<float> anchor_offset;
		std::vector<float> anchor_feature;
		std::vector<float> anchor_opacity;
		std::vector<float> anchor_scale_1;
		std::vector<float> anchor_scale_2;
		std::vector<float> anchor_rotation;
		std::vector<float> gaussian_pos;

		torch::Tensor ak_pos_all;
		torch::Tensor ak_feat_all;
		torch::Tensor ak_rot_all;
		torch::Tensor ak_scale_1;
		torch::Tensor ak_scale_2;
		torch::Tensor gs_pos_all;

		//float* pos_cuda;
		//float* rot_cuda;
		//float* scale_cuda;
		//float* opacity_cuda;
		//float* color_cuda;
		int* rect_cuda;

		GLuint imageBuffer;
		cudaGraphicsResource_t imageBufferCuda;

		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		size_t allocdGeom_filter = 0, allocdBinning_filter = 0, allocdImg_filter = 0;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
		void* geomPtr_filter = nullptr, * binningPtr_filter = nullptr, * imgPtr_filter = nullptr;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;
		std::function<char* (size_t N)> geomBufferFunc_filter, binningBufferFunc_filter, imgBufferFunc_filter;

		float* view_cuda;
		float* proj_cuda;
		float* cam_pos_cuda;
		float* background_cuda;

		float _scalingModifier = 1.0f;
		GaussianData* gData;

		bool _interop_failed = false;
		std::vector<char> fallback_bytes;
		float* fallbackBufferCuda = nullptr;
		bool accepted = false;


		std::shared_ptr<sibr::BasicIBRScene> _scene; ///< The current scene.
		PointBasedRenderer::Ptr _pointbasedrenderer;
		BufferCopyRenderer* _copyRenderer;
		GaussianSurfaceRenderer* _gaussianRenderer;
	};

} /*namespace sibr*/
