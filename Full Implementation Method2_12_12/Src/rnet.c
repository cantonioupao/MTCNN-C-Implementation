/**
  ******************************************************************************
  * @file    rnet.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Mon Jun  1 01:24:31 2020
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


#include "rnet.h"

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
#define AI_NET_OBJ_INSTANCE g_rnet
 
#undef AI_RNET_MODEL_SIGNATURE
#define AI_RNET_MODEL_SIGNATURE     "ae029186839ecc1c249695ba441d8715"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-5.0.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Mon Jun  1 01:24:31 2020"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_RNET_N_BATCHES
#define AI_RNET_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array conv2_scratch0_array;   /* Array #0 */
AI_STATIC ai_array conv1_scratch0_array;   /* Array #1 */
AI_STATIC ai_array conv52_bias_array;   /* Array #2 */
AI_STATIC ai_array conv52_weights_array;   /* Array #3 */
AI_STATIC ai_array conv51_bias_array;   /* Array #4 */
AI_STATIC ai_array conv51_weights_array;   /* Array #5 */
AI_STATIC ai_array prelu4_alpha_array;   /* Array #6 */
AI_STATIC ai_array conv4_bias_array;   /* Array #7 */
AI_STATIC ai_array conv4_weights_array;   /* Array #8 */
AI_STATIC ai_array conv3_alpha_array;   /* Array #9 */
AI_STATIC ai_array conv3_bias_array;   /* Array #10 */
AI_STATIC ai_array conv3_weights_array;   /* Array #11 */
AI_STATIC ai_array conv2_alpha_array;   /* Array #12 */
AI_STATIC ai_array conv2_bias_array;   /* Array #13 */
AI_STATIC ai_array conv2_weights_array;   /* Array #14 */
AI_STATIC ai_array conv1_alpha_array;   /* Array #15 */
AI_STATIC ai_array conv1_bias_array;   /* Array #16 */
AI_STATIC ai_array conv1_weights_array;   /* Array #17 */
AI_STATIC ai_array input_10_output_array;   /* Array #18 */
AI_STATIC ai_array conv1_output_array;   /* Array #19 */
AI_STATIC ai_array conv2_output_array;   /* Array #20 */
AI_STATIC ai_array conv3_output_array;   /* Array #21 */
AI_STATIC ai_array permute_6_output_array;   /* Array #22 */
AI_STATIC ai_array conv4_output_array;   /* Array #23 */
AI_STATIC ai_array prelu4_output_array;   /* Array #24 */
AI_STATIC ai_array conv52_output_array;   /* Array #25 */
AI_STATIC ai_array conv51_output_array;   /* Array #26 */
AI_STATIC ai_array conv51_nl_output_array;   /* Array #27 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor conv2_scratch0;   /* Tensor #0 */
AI_STATIC ai_tensor conv1_scratch0;   /* Tensor #1 */
AI_STATIC ai_tensor conv52_bias;   /* Tensor #2 */
AI_STATIC ai_tensor conv52_weights;   /* Tensor #3 */
AI_STATIC ai_tensor conv51_bias;   /* Tensor #4 */
AI_STATIC ai_tensor conv51_weights;   /* Tensor #5 */
AI_STATIC ai_tensor prelu4_alpha;   /* Tensor #6 */
AI_STATIC ai_tensor conv4_bias;   /* Tensor #7 */
AI_STATIC ai_tensor conv4_weights;   /* Tensor #8 */
AI_STATIC ai_tensor conv3_alpha;   /* Tensor #9 */
AI_STATIC ai_tensor conv3_bias;   /* Tensor #10 */
AI_STATIC ai_tensor conv3_weights;   /* Tensor #11 */
AI_STATIC ai_tensor conv2_alpha;   /* Tensor #12 */
AI_STATIC ai_tensor conv2_bias;   /* Tensor #13 */
AI_STATIC ai_tensor conv2_weights;   /* Tensor #14 */
AI_STATIC ai_tensor conv1_alpha;   /* Tensor #15 */
AI_STATIC ai_tensor conv1_bias;   /* Tensor #16 */
AI_STATIC ai_tensor conv1_weights;   /* Tensor #17 */
AI_STATIC ai_tensor input_10_output;   /* Tensor #18 */
AI_STATIC ai_tensor conv1_output;   /* Tensor #19 */
AI_STATIC ai_tensor conv2_output;   /* Tensor #20 */
AI_STATIC ai_tensor conv3_output;   /* Tensor #21 */
AI_STATIC ai_tensor permute_6_output;   /* Tensor #22 */
AI_STATIC ai_tensor permute_6_output0;   /* Tensor #23 */
AI_STATIC ai_tensor conv4_output;   /* Tensor #24 */
AI_STATIC ai_tensor prelu4_output;   /* Tensor #25 */
AI_STATIC ai_tensor conv52_output;   /* Tensor #26 */
AI_STATIC ai_tensor conv51_output;   /* Tensor #27 */
AI_STATIC ai_tensor conv51_nl_output;   /* Tensor #28 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain conv1_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain conv2_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain conv3_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain permute_6_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain conv4_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain prelu4_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain conv52_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain conv51_chain;   /* Chain #7 */
AI_STATIC_CONST ai_tensor_chain conv51_nl_chain;   /* Chain #8 */


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_conv2d_nl_pool conv1_layer; /* Layer #0 */
AI_STATIC ai_layer_conv2d_nl_pool conv2_layer; /* Layer #1 */
AI_STATIC ai_layer_conv2d conv3_layer; /* Layer #2 */
AI_STATIC ai_layer_transpose permute_6_layer; /* Layer #3 */
AI_STATIC ai_layer_dense conv4_layer; /* Layer #4 */
AI_STATIC ai_layer_nl prelu4_layer; /* Layer #5 */
AI_STATIC ai_layer_dense conv52_layer; /* Layer #6 */
AI_STATIC ai_layer_dense conv51_layer; /* Layer #7 */
AI_STATIC ai_layer_nl conv51_nl_layer; /* Layer #8 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
    conv2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1296,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 1848,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv52_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 4,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv52_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 512,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv51_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 2,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv51_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 256,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    prelu4_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 128,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv4_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 128,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv4_weights_array, AI_ARRAY_FORMAT_LUT8_FLOAT,
    NULL, NULL, 73728,
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
    NULL, NULL, 12288,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 48,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 48,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 12096,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_alpha_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 28,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 28,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 756,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_10_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 1728,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv1_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 3388,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 768,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv3_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 576,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    permute_6_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 576,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv4_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 128,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    prelu4_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 128,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv52_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 4,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv51_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 2,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv51_nl_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 2,
     AI_STATIC)




/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
  conv2_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 48, 9, 3), AI_STRIDE_INIT(4, 4, 4, 192, 1728),
  1, &conv2_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 28, 22, 3), AI_STRIDE_INIT(4, 4, 4, 112, 2464),
  1, &conv1_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv52_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv52_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv52_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 4, 1, 1), AI_STRIDE_INIT(4, 4, 512, 2048, 2048),
  1, &conv52_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv51_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &conv51_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv51_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 2, 1, 1), AI_STRIDE_INIT(4, 4, 512, 1024, 1024),
  1, &conv51_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  prelu4_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &prelu4_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv4_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv4_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv4_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 576, 128, 1, 1), AI_STRIDE_INIT(4, 1, 576, 73728, 73728),
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
  0x0, 0x0, AI_SHAPE_INIT(4, 48, 2, 2, 64), AI_STRIDE_INIT(4, 4, 192, 384, 768),
  1, &conv3_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &conv2_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &conv2_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 28, 3, 3, 48), AI_STRIDE_INIT(4, 4, 112, 336, 1008),
  1, &conv2_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_alpha, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 28, 1, 1), AI_STRIDE_INIT(4, 4, 4, 112, 112),
  1, &conv1_alpha_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 28, 1, 1), AI_STRIDE_INIT(4, 4, 4, 112, 112),
  1, &conv1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 3, 3, 3, 28), AI_STRIDE_INIT(4, 4, 12, 36, 108),
  1, &conv1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  input_10_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 24, 24), AI_STRIDE_INIT(4, 4, 4, 12, 288),
  1, &input_10_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 28, 11, 11), AI_STRIDE_INIT(4, 4, 4, 112, 1232),
  1, &conv1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 48, 4, 4), AI_STRIDE_INIT(4, 4, 4, 192, 768),
  1, &conv2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv3_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 3, 3), AI_STRIDE_INIT(4, 4, 4, 256, 768),
  1, &conv3_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  permute_6_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 4, 4, 12, 36),
  1, &permute_6_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  permute_6_output0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 576, 1, 1), AI_STRIDE_INIT(4, 4, 4, 2304, 2304),
  1, &permute_6_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv4_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv4_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  prelu4_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &prelu4_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv52_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
  1, &conv52_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv51_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &conv51_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv51_nl_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &conv51_nl_output_array, NULL)


/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&input_10_output),
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
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv3_layer, 7,
  CONV2D_TYPE,
  conv2d, forward_conv2d,
  &AI_NET_OBJ_INSTANCE, &permute_6_layer, AI_STATIC,
  .tensors = &conv3_chain, 
  .groups = 1, 
  .nl_func = nl_func_prelu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  permute_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv3_output),
  AI_TENSOR_LIST_ENTRY(&permute_6_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  permute_6_layer, 9,
  TRANSPOSE_TYPE,
  transpose, forward_transpose,
  &AI_NET_OBJ_INSTANCE, &conv4_layer, AI_STATIC,
  .tensors = &permute_6_chain, 
  .out_mapping = AI_SHAPE_INIT(4, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&permute_6_output0),
  AI_TENSOR_LIST_ENTRY(&conv4_output),
  AI_TENSOR_LIST_ENTRY(&conv4_weights, &conv4_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv4_layer, 11,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &prelu4_layer, AI_STATIC,
  .tensors = &conv4_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  prelu4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv4_output),
  AI_TENSOR_LIST_ENTRY(&prelu4_output),
  AI_TENSOR_LIST_ENTRY(&prelu4_alpha),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  prelu4_layer, 12,
  NL_TYPE,
  nl, forward_prelu,
  &AI_NET_OBJ_INSTANCE, &conv52_layer, AI_STATIC,
  .tensors = &prelu4_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv52_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&prelu4_output),
  AI_TENSOR_LIST_ENTRY(&conv52_output),
  AI_TENSOR_LIST_ENTRY(&conv52_weights, &conv52_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv52_layer, 14,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &conv51_layer, AI_STATIC,
  .tensors = &conv52_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv51_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&prelu4_output),
  AI_TENSOR_LIST_ENTRY(&conv51_output),
  AI_TENSOR_LIST_ENTRY(&conv51_weights, &conv51_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv51_layer, 13,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &conv51_nl_layer, AI_STATIC,
  .tensors = &conv51_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv51_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv51_output),
  AI_TENSOR_LIST_ENTRY(&conv51_nl_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv51_nl_layer, 13,
  NL_TYPE,
  nl, forward_sm,
  &AI_NET_OBJ_INSTANCE, &conv51_nl_layer, AI_STATIC,
  .tensors = &conv51_nl_chain, 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 180552, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 21808, 1,
                     NULL),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_RNET_IN_NUM, &input_10_output),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_RNET_OUT_NUM, &conv51_nl_output, &conv52_output),
  &conv1_layer, 0, NULL)



AI_DECLARE_STATIC
ai_bool rnet_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
  AI_ASSERT(activations)
  AI_UNUSED(net_ctx)

  {
    /* Updating activations (byte) offsets */
    conv2_scratch0_array.data = AI_PTR(activations + 0);
    conv2_scratch0_array.data_start = AI_PTR(activations + 0);
    conv1_scratch0_array.data = AI_PTR(activations + 0);
    conv1_scratch0_array.data_start = AI_PTR(activations + 0);
    input_10_output_array.data = AI_PTR(NULL);
    input_10_output_array.data_start = AI_PTR(NULL);
    conv1_output_array.data = AI_PTR(activations + 8256);
    conv1_output_array.data_start = AI_PTR(activations + 8256);
    conv2_output_array.data = AI_PTR(activations + 5184);
    conv2_output_array.data_start = AI_PTR(activations + 5184);
    conv3_output_array.data = AI_PTR(activations + 0);
    conv3_output_array.data_start = AI_PTR(activations + 0);
    permute_6_output_array.data = AI_PTR(activations + 2304);
    permute_6_output_array.data_start = AI_PTR(activations + 2304);
    conv4_output_array.data = AI_PTR(activations + 0);
    conv4_output_array.data_start = AI_PTR(activations + 0);
    prelu4_output_array.data = AI_PTR(activations + 512);
    prelu4_output_array.data_start = AI_PTR(activations + 512);
    conv52_output_array.data = AI_PTR(NULL);
    conv52_output_array.data_start = AI_PTR(NULL);
    conv51_output_array.data = AI_PTR(activations + 0);
    conv51_output_array.data_start = AI_PTR(activations + 0);
    conv51_nl_output_array.data = AI_PTR(NULL);
    conv51_nl_output_array.data_start = AI_PTR(NULL);
    
  }
  return true;
}



AI_DECLARE_STATIC
ai_bool rnet_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT(weights)
  AI_UNUSED(net_ctx)

  {
    /* Updating weights (byte) offsets */
    
    conv52_bias_array.format |= AI_FMT_FLAG_CONST;
    conv52_bias_array.data = AI_PTR(weights + 180536);
    conv52_bias_array.data_start = AI_PTR(weights + 180536);
    conv52_weights_array.format |= AI_FMT_FLAG_CONST;
    conv52_weights_array.data = AI_PTR(weights + 178488);
    conv52_weights_array.data_start = AI_PTR(weights + 178488);
    conv51_bias_array.format |= AI_FMT_FLAG_CONST;
    conv51_bias_array.data = AI_PTR(weights + 178480);
    conv51_bias_array.data_start = AI_PTR(weights + 178480);
    conv51_weights_array.format |= AI_FMT_FLAG_CONST;
    conv51_weights_array.data = AI_PTR(weights + 177456);
    conv51_weights_array.data_start = AI_PTR(weights + 177456);
    prelu4_alpha_array.format |= AI_FMT_FLAG_CONST;
    prelu4_alpha_array.data = AI_PTR(weights + 176944);
    prelu4_alpha_array.data_start = AI_PTR(weights + 176944);
    conv4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv4_bias_array.data = AI_PTR(weights + 176432);
    conv4_bias_array.data_start = AI_PTR(weights + 176432);
    conv4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv4_weights_array.data = AI_PTR(weights + 102704);
    conv4_weights_array.data_start = AI_PTR(weights + 101680);
    conv3_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv3_alpha_array.data = AI_PTR(weights + 101424);
    conv3_alpha_array.data_start = AI_PTR(weights + 101424);
    conv3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv3_bias_array.data = AI_PTR(weights + 101168);
    conv3_bias_array.data_start = AI_PTR(weights + 101168);
    conv3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv3_weights_array.data = AI_PTR(weights + 52016);
    conv3_weights_array.data_start = AI_PTR(weights + 52016);
    conv2_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv2_alpha_array.data = AI_PTR(weights + 51824);
    conv2_alpha_array.data_start = AI_PTR(weights + 51824);
    conv2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2_bias_array.data = AI_PTR(weights + 51632);
    conv2_bias_array.data_start = AI_PTR(weights + 51632);
    conv2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2_weights_array.data = AI_PTR(weights + 3248);
    conv2_weights_array.data_start = AI_PTR(weights + 3248);
    conv1_alpha_array.format |= AI_FMT_FLAG_CONST;
    conv1_alpha_array.data = AI_PTR(weights + 3136);
    conv1_alpha_array.data_start = AI_PTR(weights + 3136);
    conv1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv1_bias_array.data = AI_PTR(weights + 3024);
    conv1_bias_array.data_start = AI_PTR(weights + 3024);
    conv1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv1_weights_array.data = AI_PTR(weights + 0);
    conv1_weights_array.data_start = AI_PTR(weights + 0);
  }

  return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_rnet_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_RNET_MODEL_NAME,
      .model_signature   = AI_RNET_MODEL_SIGNATURE,
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
      
      .n_macc            = 1604630,
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
ai_error ai_rnet_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_rnet_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_rnet_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_rnet_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= rnet_configure_weights(net_ctx, &params->params);
  ok &= rnet_configure_activations(net_ctx, &params->activations);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_rnet_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_rnet_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}

#undef AI_RNET_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

