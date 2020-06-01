/**
  ******************************************************************************
  * @file    onet.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Jun  1 01:24:39 2020
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


#include "onet.h"

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
#define AI_NET_OBJ_INSTANCE g_onet
 
#undef AI_ONET_MODEL_SIGNATURE
#define AI_ONET_MODEL_SIGNATURE     "0fed9e69953bbe9eb1ec18f0b5067615"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-5.0.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Mon Jun  1 01:24:39 2020"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_ONET_N_BATCHES
#define AI_ONET_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array conv3_scratch0_array;   /* Array #0 */
AI_STATIC ai_array conv2_scratch0_array;   /* Array #1 */
AI_STATIC ai_array conv1_scratch0_array;   /* Array #2 */
AI_STATIC ai_array conv62_bias_array;   /* Array #3 */
AI_STATIC ai_array conv62_weights_array;   /* Array #4 */
AI_STATIC ai_array prelu5_alpha_array;   /* Array #5 */
AI_STATIC ai_array conv5_bias_array;   /* Array #6 */
AI_STATIC ai_array conv5_weights_array;   /* Array #7 */
AI_STATIC ai_array conv4_alpha_array;   /* Array #8 */
AI_STATIC ai_array conv4_bias_array;   /* Array #9 */
AI_STATIC ai_array conv4_weights_array;   /* Array #10 */
AI_STATIC ai_array conv3_alpha_array;   /* Array #11 */
AI_STATIC ai_array conv3_bias_array;   /* Array #12 */
AI_STATIC ai_array conv3_weights_array;   /* Array #13 */
AI_STATIC ai_array conv2_alpha_array;   /* Array #14 */
AI_STATIC ai_array conv2_bias_array;   /* Array #15 */
AI_STATIC ai_array conv2_weights_array;   /* Array #16 */
AI_STATIC ai_array conv1_alpha_array;   /* Array #17 */
AI_STATIC ai_array conv1_bias_array;   /* Array #18 */
AI_STATIC ai_array conv1_weights_array;   /* Array #19 */
AI_STATIC ai_array input_2_output_array;   /* Array #20 */
AI_STATIC ai_array conv1_output_array;   /* Array #21 */
AI_STATIC ai_array conv2_output_array;   /* Array #22 */
AI_STATIC ai_array conv3_output_array;   /* Array #23 */
AI_STATIC ai_array conv4_output_array;   /* Array #24 */
AI_STATIC ai_array permute_2_output_array;   /* Array #25 */
AI_STATIC ai_array conv5_output_array;   /* Array #26 */
AI_STATIC ai_array prelu5_output_array;   /* Array #27 */
AI_STATIC ai_array conv62_output_array;   /* Array #28 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor conv3_scratch0;   /* Tensor #0 */
AI_STATIC ai_tensor conv2_scratch0;   /* Tensor #1 */
AI_STATIC ai_tensor conv1_scratch0;   /* Tensor #2 */
AI_STATIC ai_tensor conv62_bias;   /* Tensor #3 */
AI_STATIC ai_tensor conv62_weights;   /* Tensor #4 */
AI_STATIC ai_tensor prelu5_alpha;   /* Tensor #5 */
AI_STATIC ai_tensor conv5_bias;   /* Tensor #6 */
AI_STATIC ai_tensor conv5_weights;   /* Tensor #7 */
AI_STATIC ai_tensor conv4_alpha;   /* Tensor #8 */
AI_STATIC ai_tensor conv4_bias;   /* Tensor #9 */
AI_STATIC ai_tensor conv4_weights;   /* Tensor #10 */
AI_STATIC ai_tensor conv3_alpha;   /* Tensor #11 */
AI_STATIC ai_tensor conv3_bias;   /* Tensor #12 */
AI_STATIC ai_tensor conv3_weights;   /* Tensor #13 */
AI_STATIC ai_tensor conv2_alpha;   /* Tensor #14 */
AI_STATIC ai_tensor conv2_bias;   /* Tensor #15 */
AI_STATIC ai_tensor conv2_weights;   /* Tensor #16 */
AI_STATIC ai_tensor conv1_alpha;   /* Tensor #17 */
AI_STATIC ai_tensor conv1_bias;   /* Tensor #18 */
AI_STATIC ai_tensor conv1_weights;   /* Tensor #19 */
AI_STATIC ai_tensor input_2_output;   /* Tensor #20 */
AI_STATIC ai_tensor conv1_output;   /* Tensor #21 */
AI_STATIC ai_tensor conv2_output;   /* Tensor #22 */
AI_STATIC ai_tensor conv3_output;   /* Tensor #23 */
AI_STATIC ai_tensor conv4_output;   /* Tensor #24 */
AI_STATIC ai_tensor permute_2_output;   /* Tensor #25 */
AI_STATIC ai_tensor permute_2_output0;   /* Tensor #26 */
AI_STATIC ai_tensor conv5_output;   /* Tensor #27 */
AI_STATIC ai_tensor prelu5_output;   /* Tensor #28 */
AI_STATIC ai_tensor conv62_output;   /* Tensor #29 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain conv1_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain conv2_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain conv3_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain conv4_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain permute_2_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain conv5_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain prelu5_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain conv62_chain;   /* Chain #7 */


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_conv2d_nl_pool conv1_layer; /* Layer #0 */
AI_STATIC ai_layer_conv2d_nl_pool conv2_layer; /* Layer #1 */
AI_STATIC ai_layer_conv2d_nl_pool conv3_layer; /* Layer #2 */
AI_STATIC ai_layer_conv2d conv4_layer; /* Layer #3 */
AI_STATIC ai_layer_transpose permute_2_layer; /* Layer #4 */
AI_STATIC ai_layer_dense conv5_layer; /* Layer #5 */
AI_STATIC ai_layer_nl prelu5_layer; /* Layer #6 */
AI_STATIC ai_layer_dense conv62_layer; /* Layer #7 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
    conv3_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1024,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 4032,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 4416,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv62_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 4,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv62_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1024,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    prelu5_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 256,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv5_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 256,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv5_weights_array, AI_ARRAY_FORMAT_LUT8_FLOAT,
    NULL, NULL, 294912,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv4_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 128,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv4_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 128,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv4_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 32768,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv3_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 64,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv3_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 64,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv3_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 36864,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 64,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 64,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 18432,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 32,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 32,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 864,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_2_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 6912,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 16928,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 6400,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv3_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1024,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv4_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1152,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    permute_2_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1152,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv5_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 256,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    prelu5_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 256,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv62_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 4,
     AI_STATIC)




/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
  conv3_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 8, 2), AI_STRIDE_INIT(4, 4, 4, 256, 2048),
  1, &conv3_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 21, 3), AI_STRIDE_INIT(4, 4, 4, 256, 5376),
  1, &conv2_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 46, 3), AI_STRIDE_INIT(4, 4, 4, 128, 5888),
  1, &conv1_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv62_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv62_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv62_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 256, 4, 1, 1), AI_STRIDE_INIT(4, 4, 1024, 4096, 4096),
  1, &conv62_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  prelu5_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &prelu5_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv5_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv5_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv5_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1152, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1152, 294912, 294912),
  1, &conv5_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv4_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv4_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv4_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv4_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv4_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 64, 2, 2, 128), AI_STRIDE_INIT(4, 4, 256, 512, 1024),
  1, &conv4_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv3_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv3_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv3_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv3_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv3_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 4, 256, 768, 2304),
  1, &conv3_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 4, 128, 384, 1152),
  1, &conv2_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv1_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 3, 3, 3, 32), AI_STRIDE_INIT(4, 4, 12, 36, 108),
  1, &conv1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  input_2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 48, 48), AI_STRIDE_INIT(4, 4, 4, 12, 576),
  1, &input_2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 23, 23), AI_STRIDE_INIT(4, 4, 4, 128, 2944),
  1, &conv1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 4, 4, 256, 2560),
  1, &conv2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv3_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 4, 4), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &conv3_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv4_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 3, 3), AI_STRIDE_INIT(4, 4, 4, 512, 1536),
  1, &conv4_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  permute_2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 3, 128), AI_STRIDE_INIT(4, 4, 4, 12, 36),
  1, &permute_2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  permute_2_output0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1152, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4608, 4608),
  1, &permute_2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv5_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv5_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  prelu5_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &prelu5_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv62_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv62_output_array, NULL)


/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&input_2_output),
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
  .pool_size = AI_SHAPE_2D_INIT(3, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv1_output),
  AI_TENSOR_LIST_ENTRY(&conv2_output),
  AI_TENSOR_LIST_ENTRY(&conv2_weights, &conv2_bias, &conv2_alpha),
  AI_TENSOR_LIST_ENTRY(&conv2_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2_layer, 4,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &conv3_layer, AI_STATIC,
  .tensors = &conv2_chain, 
  .groups = 1, 
  .nl_func = nl_func_prelu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(3, 3), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_mp_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2_output),
  AI_TENSOR_LIST_ENTRY(&conv3_output),
  AI_TENSOR_LIST_ENTRY(&conv3_weights, &conv3_bias, &conv3_alpha),
  AI_TENSOR_LIST_ENTRY(&conv3_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv3_layer, 7,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &conv4_layer, AI_STATIC,
  .tensors = &conv3_chain, 
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
  conv4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv3_output),
  AI_TENSOR_LIST_ENTRY(&conv4_output),
  AI_TENSOR_LIST_ENTRY(&conv4_weights, &conv4_bias, &conv4_alpha),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv4_layer, 10,
  CONV2D_TYPE,
  conv2d, forward_conv2d,
  &AI_NET_OBJ_INSTANCE, &permute_2_layer, AI_STATIC,
  .tensors = &conv4_chain, 
  .groups = 1, 
  .nl_func = nl_func_prelu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  permute_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv4_output),
  AI_TENSOR_LIST_ENTRY(&permute_2_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  permute_2_layer, 12,
  TRANSPOSE_TYPE,
  transpose, forward_transpose,
  &AI_NET_OBJ_INSTANCE, &conv5_layer, AI_STATIC,
  .tensors = &permute_2_chain, 
  .out_mapping = AI_SHAPE_INIT(4, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&permute_2_output0),
  AI_TENSOR_LIST_ENTRY(&conv5_output),
  AI_TENSOR_LIST_ENTRY(&conv5_weights, &conv5_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv5_layer, 14,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &prelu5_layer, AI_STATIC,
  .tensors = &conv5_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  prelu5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv5_output),
  AI_TENSOR_LIST_ENTRY(&prelu5_output),
  AI_TENSOR_LIST_ENTRY(&prelu5_alpha),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  prelu5_layer, 15,
  NL_TYPE,
  nl, forward_prelu,
  &AI_NET_OBJ_INSTANCE, &conv62_layer, AI_STATIC,
  .tensors = &prelu5_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv62_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&prelu5_output),
  AI_TENSOR_LIST_ENTRY(&conv62_output),
  AI_TENSOR_LIST_ENTRY(&conv62_weights, &conv62_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv62_layer, 16,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &conv62_layer, AI_STATIC,
  .tensors = &conv62_chain, 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 660112, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 88192, 1,
                     NULL),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_ONET_IN_NUM, &input_2_output),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_ONET_OUT_NUM, &conv62_output),
  &conv1_layer, 0, NULL)



AI_DECLARE_STATIC
ai_bool onet_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
  AI_ASSERT(activations)
  AI_UNUSED(net_ctx)

  {
    /* Updating activations (byte) offsets */
    conv3_scratch0_array.data = AI_PTR(activations + 70528);
    conv3_scratch0_array.data_start = AI_PTR(activations + 70528);
    conv2_scratch0_array.data = AI_PTR(activations + 70528);
    conv2_scratch0_array.data_start = AI_PTR(activations + 70528);
    conv1_scratch0_array.data = AI_PTR(activations + 70528);
    conv1_scratch0_array.data_start = AI_PTR(activations + 70528);
    input_2_output_array.data = AI_PTR(NULL);
    input_2_output_array.data_start = AI_PTR(NULL);
    conv1_output_array.data = AI_PTR(activations + 2816);
    conv1_output_array.data_start = AI_PTR(activations + 2816);
    conv2_output_array.data = AI_PTR(activations + 0);
    conv2_output_array.data_start = AI_PTR(activations + 0);
    conv3_output_array.data = AI_PTR(activations + 66432);
    conv3_output_array.data_start = AI_PTR(activations + 66432);
    conv4_output_array.data = AI_PTR(activations + 63872);
    conv4_output_array.data_start = AI_PTR(activations + 63872);
    permute_2_output_array.data = AI_PTR(activations + 59264);
    permute_2_output_array.data_start = AI_PTR(activations + 59264);
    conv5_output_array.data = AI_PTR(activations + 58240);
    conv5_output_array.data_start = AI_PTR(activations + 58240);
    prelu5_output_array.data = AI_PTR(activations + 58240);
    prelu5_output_array.data_start = AI_PTR(activations + 58240);
    conv62_output_array.data = AI_PTR(NULL);
    conv62_output_array.data_start = AI_PTR(NULL);
    
  }
  return true;
}



AI_DECLARE_STATIC
ai_bool onet_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT(weights)
  AI_UNUSED(net_ctx)

  {
    /* Updating weights (byte) offsets */
    
    conv62_bias_array.format |= AI_FMT_FLAG_CONST;
    conv62_bias_array.data = AI_PTR(weights + 660096);
    conv62_bias_array.data_start = AI_PTR(weights + 660096);
    conv62_weights_array.format |= AI_FMT_FLAG_CONST;
    conv62_weights_array.data = AI_PTR(weights + 656000);
    conv62_weights_array.data_start = AI_PTR(weights + 656000);
    prelu5_alpha_array.format |= AI_FMT_FLAG_CONST;
    prelu5_alpha_array.data = AI_PTR(weights + 654976);
    prelu5_alpha_array.data_start = AI_PTR(weights + 654976);
    conv5_bias_array.format |= AI_FMT_FLAG_CONST;
    conv5_bias_array.data = AI_PTR(weights + 653952);
    conv5_bias_array.data_start = AI_PTR(weights + 653952);
    conv5_weights_array.format |= AI_FMT_FLAG_CONST;
    conv5_weights_array.data = AI_PTR(weights + 359040);
    conv5_weights_array.data_start = AI_PTR(weights + 358016);
    conv4_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv4_alpha_array.data = AI_PTR(weights + 357504);
    conv4_alpha_array.data_start = AI_PTR(weights + 357504);
    conv4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv4_bias_array.data = AI_PTR(weights + 356992);
    conv4_bias_array.data_start = AI_PTR(weights + 356992);
    conv4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv4_weights_array.data = AI_PTR(weights + 225920);
    conv4_weights_array.data_start = AI_PTR(weights + 225920);
    conv3_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv3_alpha_array.data = AI_PTR(weights + 225664);
    conv3_alpha_array.data_start = AI_PTR(weights + 225664);
    conv3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv3_bias_array.data = AI_PTR(weights + 225408);
    conv3_bias_array.data_start = AI_PTR(weights + 225408);
    conv3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv3_weights_array.data = AI_PTR(weights + 77952);
    conv3_weights_array.data_start = AI_PTR(weights + 77952);
    conv2_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv2_alpha_array.data = AI_PTR(weights + 77696);
    conv2_alpha_array.data_start = AI_PTR(weights + 77696);
    conv2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2_bias_array.data = AI_PTR(weights + 77440);
    conv2_bias_array.data_start = AI_PTR(weights + 77440);
    conv2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2_weights_array.data = AI_PTR(weights + 3712);
    conv2_weights_array.data_start = AI_PTR(weights + 3712);
    conv1_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv1_alpha_array.data = AI_PTR(weights + 3584);
    conv1_alpha_array.data_start = AI_PTR(weights + 3584);
    conv1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv1_bias_array.data = AI_PTR(weights + 3456);
    conv1_bias_array.data_start = AI_PTR(weights + 3456);
    conv1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv1_weights_array.data = AI_PTR(weights + 0);
    conv1_weights_array.data_start = AI_PTR(weights + 0);
  }

  return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_onet_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_ONET_MODEL_NAME,
      .model_signature   = AI_ONET_MODEL_SIGNATURE,
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
      
      .n_macc            = 13324096,
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
ai_error ai_onet_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_onet_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_onet_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_onet_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= onet_configure_weights(net_ctx, &params->params);
  ok &= onet_configure_activations(net_ctx, &params->activations);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_onet_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_onet_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}

#undef AI_ONET_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

