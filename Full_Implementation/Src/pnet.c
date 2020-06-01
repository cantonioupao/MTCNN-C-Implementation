/**
  ******************************************************************************
  * @file    pnet.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Sat May 30 17:57:45 2020
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2018 STMicroelectronics.
  * All rights reserved.
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */


#include "pnet.h"

#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "layers.h"

#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#define AI_TOOLS_VERSION_MAJOR 5
#define AI_TOOLS_VERSION_MINOR 0
#define AI_TOOLS_VERSION_MICRO 0


#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MAJOR 1
#define AI_TOOLS_API_VERSION_MINOR 3
#define AI_TOOLS_API_VERSION_MICRO 0

#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_pnet
 
#undef AI_PNET_MODEL_SIGNATURE
#define AI_PNET_MODEL_SIGNATURE     "61b58b0d4bbf10226f70c6b51208bd0c"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-5.0.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Sat May 30 17:57:45 2020"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_PNET_N_BATCHES
#define AI_PNET_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array conv1_scratch0_array;   /* Array #0 */
AI_STATIC ai_array conv41_bias_array;   /* Array #1 */
AI_STATIC ai_array conv41_weights_array;   /* Array #2 */
AI_STATIC ai_array conv3_alpha_array;   /* Array #3 */
AI_STATIC ai_array conv3_bias_array;   /* Array #4 */
AI_STATIC ai_array conv3_weights_array;   /* Array #5 */
AI_STATIC ai_array conv2_alpha_array;   /* Array #6 */
AI_STATIC ai_array conv2_bias_array;   /* Array #7 */
AI_STATIC ai_array conv2_weights_array;   /* Array #8 */
AI_STATIC ai_array conv1_alpha_array;   /* Array #9 */
AI_STATIC ai_array conv1_bias_array;   /* Array #10 */
AI_STATIC ai_array conv1_weights_array;   /* Array #11 */
AI_STATIC ai_array input_13_output_array;   /* Array #12 */
AI_STATIC ai_array conv1_output_array;   /* Array #13 */
AI_STATIC ai_array conv2_output_array;   /* Array #14 */
AI_STATIC ai_array conv3_output_array;   /* Array #15 */
AI_STATIC ai_array conv41_output_array;   /* Array #16 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor conv1_scratch0;   /* Tensor #0 */
AI_STATIC ai_tensor conv41_bias;   /* Tensor #1 */
AI_STATIC ai_tensor conv41_weights;   /* Tensor #2 */
AI_STATIC ai_tensor conv3_alpha;   /* Tensor #3 */
AI_STATIC ai_tensor conv3_bias;   /* Tensor #4 */
AI_STATIC ai_tensor conv3_weights;   /* Tensor #5 */
AI_STATIC ai_tensor conv2_alpha;   /* Tensor #6 */
AI_STATIC ai_tensor conv2_bias;   /* Tensor #7 */
AI_STATIC ai_tensor conv2_weights;   /* Tensor #8 */
AI_STATIC ai_tensor conv1_alpha;   /* Tensor #9 */
AI_STATIC ai_tensor conv1_bias;   /* Tensor #10 */
AI_STATIC ai_tensor conv1_weights;   /* Tensor #11 */
AI_STATIC ai_tensor input_13_output;   /* Tensor #12 */
AI_STATIC ai_tensor conv1_output;   /* Tensor #13 */
AI_STATIC ai_tensor conv2_output;   /* Tensor #14 */
AI_STATIC ai_tensor conv3_output;   /* Tensor #15 */
AI_STATIC ai_tensor conv41_output;   /* Tensor #16 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain conv1_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain conv2_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain conv3_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain conv41_chain;   /* Chain #3 */


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_conv2d_nl_pool conv1_layer; /* Layer #0 */
AI_STATIC ai_layer_conv2d conv2_layer; /* Layer #1 */
AI_STATIC ai_layer_conv2d conv3_layer; /* Layer #2 */
AI_STATIC ai_layer_conv2d conv41_layer; /* Layer #3 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
    conv1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 500,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv41_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 2,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv41_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 64,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv3_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 32,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv3_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 32,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv3_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 4608,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 16,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 16,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1440,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 10,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 10,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 270,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_13_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 2187,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1440,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1600,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv3_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 2048,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv41_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 128,
     AI_STATIC)




/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
  conv1_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 25, 2), AI_STRIDE_INIT(4, 4, 4, 40, 1000),
  1, &conv1_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv41_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &conv41_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv41_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 32, 1, 1, 2), AI_STRIDE_INIT(4, 4, 128, 128, 128),
  1, &conv41_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv3_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv3_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv3_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv3_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv3_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 4, 64, 192, 576),
  1, &conv3_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 10, 3, 3, 16), AI_STRIDE_INIT(4, 4, 40, 120, 360),
  1, &conv2_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv1_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &conv1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 3, 3, 3, 10), AI_STRIDE_INIT(4, 4, 12, 36, 108),
  1, &conv1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  input_13_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 27, 27), AI_STRIDE_INIT(4, 4, 4, 12, 324),
  1, &input_13_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 10, 12, 12), AI_STRIDE_INIT(4, 4, 4, 40, 480),
  1, &conv1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 10, 10), AI_STRIDE_INIT(4, 4, 4, 64, 640),
  1, &conv2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv3_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &conv3_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv41_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 2, 8, 8), AI_STRIDE_INIT(4, 4, 4, 8, 64),
  1, &conv41_output_array, NULL)


/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&input_13_output),
  AI_TENSOR_LIST_ENTRY(&conv1_output),
  AI_TENSOR_LIST_ENTRY(&conv1_weights, &conv1_bias, &conv1_alpha),
  AI_TENSOR_LIST_ENTRY(&conv1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv1_layer, 1,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &conv2_layer, AI_STATIC,
  .tensors = &conv1_chain, 
  .groups = 1, 
  .nl_func = nl_func_prelu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv1_output),
  AI_TENSOR_LIST_ENTRY(&conv2_output),
  AI_TENSOR_LIST_ENTRY(&conv2_weights, &conv2_bias, &conv2_alpha),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2_layer, 4,
  CONV2D_TYPE,
  conv2d, forward_conv2d,
  &AI_NET_OBJ_INSTANCE, &conv3_layer, AI_STATIC,
  .tensors = &conv2_chain, 
  .groups = 1, 
  .nl_func = nl_func_prelu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2_output),
  AI_TENSOR_LIST_ENTRY(&conv3_output),
  AI_TENSOR_LIST_ENTRY(&conv3_weights, &conv3_bias, &conv3_alpha),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv3_layer, 6,
  CONV2D_TYPE,
  conv2d, forward_conv2d,
  &AI_NET_OBJ_INSTANCE, &conv41_layer, AI_STATIC,
  .tensors = &conv3_chain, 
  .groups = 1, 
  .nl_func = nl_func_prelu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv41_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv3_output),
  AI_TENSOR_LIST_ENTRY(&conv41_output),
  AI_TENSOR_LIST_ENTRY(&conv41_weights, &conv41_bias, NULL),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv41_layer, 8,
  CONV2D_TYPE,
  conv2d, forward_conv2d,
  &AI_NET_OBJ_INSTANCE, &conv41_layer, AI_STATIC,
  .tensors = &conv41_chain, 
  .groups = 1, 
  .nl_func = nl_func_sm_channel_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 26000, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 12048, 1,
                     NULL),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_PNET_IN_NUM, &input_13_output),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_PNET_OUT_NUM, &conv41_output),
  &conv1_layer, 0, NULL)



AI_DECLARE_STATIC
ai_bool pnet_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
  AI_ASSERT(activations)
  AI_UNUSED(net_ctx)

  {
    /* Updating activations (byte) offsets */
    conv1_scratch0_array.data = AI_PTR(activations + 10048);
    conv1_scratch0_array.data_start = AI_PTR(activations + 10048);
    input_13_output_array.data = AI_PTR(NULL);
    input_13_output_array.data_start = AI_PTR(NULL);
    conv1_output_array.data = AI_PTR(activations + 4288);
    conv1_output_array.data_start = AI_PTR(activations + 4288);
    conv2_output_array.data = AI_PTR(activations + 2944);
    conv2_output_array.data_start = AI_PTR(activations + 2944);
    conv3_output_array.data = AI_PTR(activations + 0);
    conv3_output_array.data_start = AI_PTR(activations + 0);
    conv41_output_array.data = AI_PTR(NULL);
    conv41_output_array.data_start = AI_PTR(NULL);
    
  }
  return true;
}



AI_DECLARE_STATIC
ai_bool pnet_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT(weights)
  AI_UNUSED(net_ctx)

  {
    /* Updating weights (byte) offsets */
    
    conv41_bias_array.format |= AI_FMT_FLAG_CONST;
    conv41_bias_array.data = AI_PTR(weights + 25992);
    conv41_bias_array.data_start = AI_PTR(weights + 25992);
    conv41_weights_array.format |= AI_FMT_FLAG_CONST;
    conv41_weights_array.data = AI_PTR(weights + 25736);
    conv41_weights_array.data_start = AI_PTR(weights + 25736);
    conv3_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv3_alpha_array.data = AI_PTR(weights + 25608);
    conv3_alpha_array.data_start = AI_PTR(weights + 25608);
    conv3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv3_bias_array.data = AI_PTR(weights + 25480);
    conv3_bias_array.data_start = AI_PTR(weights + 25480);
    conv3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv3_weights_array.data = AI_PTR(weights + 7048);
    conv3_weights_array.data_start = AI_PTR(weights + 7048);
    conv2_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv2_alpha_array.data = AI_PTR(weights + 6984);
    conv2_alpha_array.data_start = AI_PTR(weights + 6984);
    conv2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2_bias_array.data = AI_PTR(weights + 6920);
    conv2_bias_array.data_start = AI_PTR(weights + 6920);
    conv2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2_weights_array.data = AI_PTR(weights + 1160);
    conv2_weights_array.data_start = AI_PTR(weights + 1160);
    conv1_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv1_alpha_array.data = AI_PTR(weights + 1120);
    conv1_alpha_array.data_start = AI_PTR(weights + 1120);
    conv1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv1_bias_array.data = AI_PTR(weights + 1080);
    conv1_bias_array.data_start = AI_PTR(weights + 1080);
    conv1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv1_weights_array.data = AI_PTR(weights + 0);
    conv1_weights_array.data_start = AI_PTR(weights + 0);
  }

  return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_pnet_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_PNET_MODEL_NAME,
      .model_signature   = AI_PNET_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = {AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR,
                            AI_TOOLS_API_VERSION_MICRO, 0x0},

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 639294,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .activations       = AI_STRUCT_INIT,
      .params            = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if ( !ai_platform_api_get_network_report(network, &r) ) return false;

    *report = r;
    return true;
  }

  return false;
}

AI_API_ENTRY
ai_error ai_pnet_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_pnet_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_pnet_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_pnet_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= pnet_configure_weights(net_ctx, &params->params);
  ok &= pnet_configure_activations(net_ctx, &params->activations);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_pnet_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_pnet_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}

#undef AI_PNET_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

