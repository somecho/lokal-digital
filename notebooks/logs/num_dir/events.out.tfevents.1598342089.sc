       ŁK"	  @ň0Ń×Abrain.Event:2ÇDuKm     b3ÁĽ	Tsň0Ń×A"žÚ
p
dense_1_inputPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
shape:˙˙˙˙˙˙˙˙˙1*
dtype0

-dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
_class
loc:@dense/kernel*
dtype0*
valueB"1      

+dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *<ž*
dtype0

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *<>*
_output_shapes
: *
dtype0*
_class
loc:@dense/kernel
ć
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_output_shapes
:	1*
seed2 *
_class
loc:@dense/kernel*

seed *
dtype0*
T0
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
: 
á
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	1*
_class
loc:@dense/kernel
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	1*
_class
loc:@dense/kernel
Ł
dense/kernel
VariableV2*
	container *
shared_name *
shape:	1*
dtype0*
_output_shapes
:	1*
_class
loc:@dense/kernel
Č
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
_output_shapes
:	1*
T0*
validate_shape(*
_class
loc:@dense/kernel
v
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	1

dense/bias/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0*
_class
loc:@dense/bias


dense/bias
VariableV2*
_output_shapes	
:*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@dense/bias
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
l
dense/bias/readIdentity
dense/bias*
_output_shapes	
:*
_class
loc:@dense/bias*
T0

dense/MatMulMatMuldense_1_inputdense/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
transpose_a( 

dense/BiasAddBiasAdddense/MatMuldense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
data_formatNHWC
T

dense/ReluReludense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *!
_class
loc:@dense_1/kernel*
_output_shapes
:*
dtype0

-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *   ž*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
valueB
 *   >*
dtype0
í
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0* 
_output_shapes
:
*

seed *
seed2 *
dtype0*!
_class
loc:@dense_1/kernel
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
ę
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel
Ü
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*!
_class
loc:@dense_1/kernel
Š
dense_1/kernel
VariableV2*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
	container *
shared_name *
shape:
*
dtype0
Ń
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel*
T0*
use_locking(
}
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0* 
_output_shapes
:


dense_1/bias/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:*
_class
loc:@dense_1/bias

dense_1/bias
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes	
:*
shared_name *
dtype0*
	container *
shape:
ť
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes	
:*
T0*
use_locking(
r
dense_1/bias/readIdentitydense_1/bias*
_output_shapes	
:*
_class
loc:@dense_1/bias*
T0

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
T0*
transpose_a( *
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
dense_2/ReluReludense_2/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_2/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_2/kernel*
_output_shapes
:*
valueB"      *
dtype0

-dense_2/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *óľ˝

-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
dtype0*
valueB
 *óľ=*
_output_shapes
: 
í
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
seed2 * 
_output_shapes
:
*
dtype0*

seed *!
_class
loc:@dense_2/kernel*
T0
Ö
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_2/kernel
ę
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*!
_class
loc:@dense_2/kernel*
T0
Ü
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:

Š
dense_2/kernel
VariableV2*
	container *
shared_name *!
_class
loc:@dense_2/kernel*
dtype0* 
_output_shapes
:
*
shape:

Ń
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(*!
_class
loc:@dense_2/kernel
}
dense_2/kernel/readIdentitydense_2/kernel* 
_output_shapes
:
*
T0*!
_class
loc:@dense_2/kernel

dense_2/bias/Initializer/zerosConst*
_class
loc:@dense_2/bias*
dtype0*
valueB*    *
_output_shapes	
:

dense_2/bias
VariableV2*
dtype0*
shared_name *
	container *
shape:*
_output_shapes	
:*
_class
loc:@dense_2/bias
ť
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*
_class
loc:@dense_2/bias
r
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes	
:

dense_3/MatMulMatMuldense_2/Reludense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_b( *
transpose_a( 

dense_3/BiasAddBiasAdddense_3/MatMuldense_2/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
X
dense_3/ReluReludense_3/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"      *!
_class
loc:@dense_3/kernel*
_output_shapes
:

-dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *żđÚ˝*
_output_shapes
: *!
_class
loc:@dense_3/kernel

-dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *żđÚ=*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0
ě
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
_output_shapes
:	*!
_class
loc:@dense_3/kernel*

seed *
seed2 *
T0*
dtype0
Ö
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_3/kernel
é
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_3/kernel*
_output_shapes
:	*
T0
Ű
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	
§
dense_3/kernel
VariableV2*
shape:	*
	container *
dtype0*
_output_shapes
:	*
shared_name *!
_class
loc:@dense_3/kernel
Đ
dense_3/kernel/AssignAssigndense_3/kernel)dense_3/kernel/Initializer/random_uniform*
T0*
_output_shapes
:	*!
_class
loc:@dense_3/kernel*
use_locking(*
validate_shape(
|
dense_3/kernel/readIdentitydense_3/kernel*
T0*
_output_shapes
:	*!
_class
loc:@dense_3/kernel

dense_3/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
_class
loc:@dense_3/bias*
dtype0

dense_3/bias
VariableV2*
_class
loc:@dense_3/bias*
dtype0*
shape:*
shared_name *
_output_shapes
:*
	container 
ş
dense_3/bias/AssignAssigndense_3/biasdense_3/bias/Initializer/zeros*
_output_shapes
:*
T0*
_class
loc:@dense_3/bias*
use_locking(*
validate_shape(
q
dense_3/bias/readIdentitydense_3/bias*
_class
loc:@dense_3/bias*
T0*
_output_shapes
:

dense_4/MatMulMatMuldense_3/Reludense_3/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0*
transpose_a( 

dense_4/BiasAddBiasAdddense_4/MatMuldense_3/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
]
dense_4/SoftmaxSoftmaxdense_4/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
Adam/iterations/initial_valueConst*
dtype0	*
value	B	 R *
_output_shapes
: 
s
Adam/iterations
VariableV2*
shape: *
_output_shapes
: *
dtype0	*
	container *
shared_name 
ž
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*"
_class
loc:@Adam/iterations*
use_locking(*
validate_shape(*
_output_shapes
: *
T0	
v
Adam/iterations/readIdentityAdam/iterations*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
Z
Adam/lr/initial_valueConst*
valueB
 *ˇŃ8*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shared_name *
shape: 

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/lr/readIdentityAdam/lr*
T0*
_class
loc:@Adam/lr*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?
o
Adam/beta_1
VariableV2*
dtype0*
_output_shapes
: *
shape: *
shared_name *
	container 
Ž
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
_class
loc:@Adam/beta_1*
validate_shape(*
T0*
_output_shapes
: *
use_locking(
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_class
loc:@Adam/beta_1*
_output_shapes
: 
^
Adam/beta_2/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *wž?
o
Adam/beta_2
VariableV2*
_output_shapes
: *
dtype0*
	container *
shared_name *
shape: 
Ž
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
use_locking(*
_class
loc:@Adam/beta_2*
T0*
_output_shapes
: *
validate_shape(
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_class
loc:@Adam/beta_2*
_output_shapes
: 
]
Adam/decay/initial_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
n

Adam/decay
VariableV2*
shared_name *
_output_shapes
: *
	container *
dtype0*
shape: 
Ş
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
T0*
validate_shape(*
_class
loc:@Adam/decay*
_output_shapes
: *
use_locking(
g
Adam/decay/readIdentity
Adam/decay*
T0*
_output_shapes
: *
_class
loc:@Adam/decay

dense_4_targetPlaceholder*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
q
dense_4_sample_weightsPlaceholder*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
\
loss/dense_4_loss/ConstConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
\
loss/dense_4_loss/sub/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
o
loss/dense_4_loss/subSubloss/dense_4_loss/sub/xloss/dense_4_loss/Const*
T0*
_output_shapes
: 

'loss/dense_4_loss/clip_by_value/MinimumMinimumdense_4/Softmaxloss/dense_4_loss/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

loss/dense_4_loss/clip_by_valueMaximum'loss/dense_4_loss/clip_by_value/Minimumloss/dense_4_loss/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
o
loss/dense_4_loss/LogLogloss/dense_4_loss/clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
loss/dense_4_loss/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

loss/dense_4_loss/ReshapeReshapedense_4_targetloss/dense_4_loss/Reshape/shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
v
loss/dense_4_loss/CastCastloss/dense_4_loss/Reshape*

DstT0	*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
!loss/dense_4_loss/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"˙˙˙˙   
 
loss/dense_4_loss/Reshape_1Reshapeloss/dense_4_loss/Log!loss/dense_4_loss/Reshape_1/shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/dense_4_loss/Cast*
T0	*
out_type0*
_output_shapes
:

Yloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss/dense_4_loss/Reshape_1loss/dense_4_loss/Cast*6
_output_shapes$
":˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
Tlabels0	*
T0
k
(loss/dense_4_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB 
î
loss/dense_4_loss/MeanMeanYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(loss/dense_4_loss/Mean/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *
T0*

Tidx0
z
loss/dense_4_loss/mulMulloss/dense_4_loss/Meandense_4_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
loss/dense_4_loss/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    

loss/dense_4_loss/NotEqualNotEqualdense_4_sample_weightsloss/dense_4_loss/NotEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
loss/dense_4_loss/Cast_1Castloss/dense_4_loss/NotEqual*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
c
loss/dense_4_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_4_loss/Mean_1Meanloss/dense_4_loss/Cast_1loss/dense_4_loss/Const_1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0

loss/dense_4_loss/truedivRealDivloss/dense_4_loss/mulloss/dense_4_loss/Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
loss/dense_4_loss/Const_2Const*
_output_shapes
:*
valueB: *
dtype0

loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_2*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
O

loss/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
V
loss/mulMul
loss/mul/xloss/dense_4_loss/Mean_2*
T0*
_output_shapes
: 
l
!metrics/acc/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

metrics/acc/MaxMaxdense_4_target!metrics/acc/Max/reduction_indices*
T0*
	keep_dims( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
g
metrics/acc/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

metrics/acc/ArgMaxArgMaxdense_4/Softmaxmetrics/acc/ArgMax/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0*
output_type0	
i
metrics/acc/CastCastmetrics/acc/ArgMax*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0	
k
metrics/acc/EqualEqualmetrics/acc/Maxmetrics/acc/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
j
metrics/acc/Cast_1Castmetrics/acc/Equal*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0

[
metrics/acc/ConstConst*
dtype0*
valueB: *
_output_shapes
:
}
metrics/acc/MeanMeanmetrics/acc/Cast_1metrics/acc/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
}
training/Adam/gradients/ShapeConst*
_class
loc:@loss/mul*
dtype0*
valueB *
_output_shapes
: 

!training/Adam/gradients/grad_ys_0Const*
_output_shapes
: *
dtype0*
_class
loc:@loss/mul*
valueB
 *  ?
¤
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_output_shapes
: *
_class
loc:@loss/mul
Ś
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_2*
_output_shapes
: *
T0*
_class
loc:@loss/mul

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_output_shapes
: *
_class
loc:@loss/mul*
T0
ş
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape/shapeConst*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_2*
dtype0*
valueB:

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape/shape*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0*
_output_shapes
:
Á
;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ShapeShapeloss/dense_4_loss/truediv*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
:*
out_type0*
T0
Ť
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape*+
_class!
loc:@loss/dense_4_loss/Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*
T0
Ă
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_1Shapeloss/dense_4_loss/truediv*+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0*
_output_shapes
:*
out_type0
­
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_2Const*
valueB *
dtype0*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
: 
˛
;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ConstConst*
valueB: *
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_2*
dtype0
Š
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const*
T0*
	keep_dims( *+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
: *

Tidx0
´
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const_1Const*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_2*
dtype0
­
<training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const_1*+
_class!
loc:@loss/dense_4_loss/Mean_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ž
?training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum/yConst*
value	B :*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_2*
dtype0

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/MaximumMaximum<training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum/y*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0

>training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum*+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0*
_output_shapes
: 
ß
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/floordiv*
_output_shapes
: *

SrcT0*

DstT0*+
_class!
loc:@loss/dense_4_loss/Mean_2

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0
ż
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
T0*
out_type0*
_output_shapes
:*,
_class"
 loc:@loss/dense_4_loss/truediv
Ż
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
: *
valueB *
dtype0
Î
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ţ
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truedivloss/dense_4_loss/Mean_1*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:*
T0
­
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0
´
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ý
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Negloss/dense_4_loss/Mean_1*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1loss/dense_4_loss/Mean_1*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulMul=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truediv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0
˝
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Ś
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
_output_shapes
: *,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*
Tshape0
¸
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean*
T0*
out_type0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
ş
:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1Shapedense_4_sample_weights*
out_type0*
T0*
_output_shapes
:*(
_class
loc:@loss/dense_4_loss/mul
ž
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul
í
6training/Adam/gradients/loss/dense_4_loss/mul_grad/mulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul
Š
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/mulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*

Tidx0*(
_class
loc:@loss/dense_4_loss/mul*
	keep_dims( *
_output_shapes
:*
T0

:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*(
_class
loc:@loss/dense_4_loss/mul
ď
8training/Adam/gradients/loss/dense_4_loss/mul_grad/mul_1Mulloss/dense_4_loss/Mean>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul
Ż
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*(
_class
loc:@loss/dense_4_loss/mul*
	keep_dims( 
Ł
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul*
T0
ý
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
out_type0*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
Ľ
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
dtype0*
value	B :
đ
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
T0*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean

7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
T0
°
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*
valueB: *
dtype0*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/deltaConst*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
dtype0*
value	B :
Ń
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*)
_class
loc:@loss/dense_4_loss/Mean*

Tidx0*
_output_shapes
:
Ť
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
value	B :

8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
T0

Atraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_4_loss/Mean_grad/range7training/Adam/gradients/loss/dense_4_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill*
N*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*)
_class
loc:@loss/dense_4_loss/Mean
Ş
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean
Ą
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/y*
T0*)
_class
loc:@loss/dense_4_loss/Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum*
T0*)
_class
loc:@loss/dense_4_loss/Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ą
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*
Tshape0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*
T0

8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*

Tmultiples0*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
˙
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_2ShapeYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0*
_output_shapes
:
ź
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*
out_type0
Ž
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0*)
_class
loc:@loss/dense_4_loss/Mean
Ą
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0*)
_class
loc:@loss/dense_4_loss/Mean
°
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
:*
valueB: 
Ľ
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( *)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*
value	B :*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0

=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/y*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
T0

>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
T0*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean
Ű
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*

SrcT0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *

DstT0

;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*)
_class
loc:@loss/dense_4_loss/Mean
Ź
"training/Adam/gradients/zeros_like	ZerosLike[loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Î
training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient[loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*´
message¨ĽCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
: 

training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivtraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
Ž
ztraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMultraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimstraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
>training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/ShapeShapeloss/dense_4_loss/Log*
T0*
out_type0*.
_class$
" loc:@loss/dense_4_loss/Reshape_1*
_output_shapes
:
÷
@training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/ReshapeReshapeztraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul>training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Shape*.
_class$
" loc:@loss/dense_4_loss/Reshape_1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

=training/Adam/gradients/loss/dense_4_loss/Log_grad/Reciprocal
Reciprocalloss/dense_4_loss/clip_by_valueA^training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/Log

6training/Adam/gradients/loss/dense_4_loss/Log_grad/mulMul@training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Reshape=training/Adam/gradients/loss/dense_4_loss/Log_grad/Reciprocal*(
_class
loc:@loss/dense_4_loss/Log*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ý
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ShapeShape'loss/dense_4_loss/clip_by_value/Minimum*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
_output_shapes
:*
out_type0*
T0
ť
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1Const*
valueB *
_output_shapes
: *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
dtype0
î
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_2Shape6training/Adam/gradients/loss/dense_4_loss/Log_grad/mul*
T0*
out_type0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
_output_shapes
:
Á
Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
Ŕ
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zerosFillDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_2Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value

Itraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_4_loss/clip_by_value/Minimumloss/dense_4_loss/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0
ć
Rtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ShapeDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
ú
Ctraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SelectSelectItraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqual6training/Adam/gradients/loss/dense_4_loss/Log_grad/mulBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0
ü
Etraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Select_1SelectItraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqualBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros6training/Adam/gradients/loss/dense_4_loss/Log_grad/mul*
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
@training/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SumSumCtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SelectRtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgs*

Tidx0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
	keep_dims( *
_output_shapes
:*
T0
É
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
Ú
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Sum_1SumEtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Select_1Ttraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgs:1*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ž
Ftraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1*
_output_shapes
: *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
Tshape0*
T0
Ő
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ShapeShapedense_4/Softmax*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:*
T0
Ë
Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1Const*
_output_shapes
: *:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
dtype0*
valueB 

Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_2ShapeDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
_output_shapes
:*
out_type0
Ń
Ptraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
dtype0
ŕ
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
ń
Ntraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualdense_4/Softmaxloss/dense_4_loss/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0

Ztraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
Ktraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SelectSelectNtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqualDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ReshapeJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Mtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Select_1SelectNtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqualJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zerosDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0
ô
Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
é
Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
ú
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0*

Tidx0
Ţ
Ntraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
ě
0training/Adam/gradients/dense_4/Softmax_grad/mulMulLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshapedense_4/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*"
_class
loc:@dense_4/Softmax
°
Btraining/Adam/gradients/dense_4/Softmax_grad/Sum/reduction_indicesConst*
dtype0*"
_class
loc:@dense_4/Softmax*
_output_shapes
:*
valueB:

0training/Adam/gradients/dense_4/Softmax_grad/SumSum0training/Adam/gradients/dense_4/Softmax_grad/mulBtraining/Adam/gradients/dense_4/Softmax_grad/Sum/reduction_indices*
	keep_dims( *
T0*"
_class
loc:@dense_4/Softmax*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
Ż
:training/Adam/gradients/dense_4/Softmax_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*"
_class
loc:@dense_4/Softmax*
valueB"˙˙˙˙   

4training/Adam/gradients/dense_4/Softmax_grad/ReshapeReshape0training/Adam/gradients/dense_4/Softmax_grad/Sum:training/Adam/gradients/dense_4/Softmax_grad/Reshape/shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*"
_class
loc:@dense_4/Softmax*
T0

0training/Adam/gradients/dense_4/Softmax_grad/subSubLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshape4training/Adam/gradients/dense_4/Softmax_grad/Reshape*
T0*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ň
2training/Adam/gradients/dense_4/Softmax_grad/mul_1Mul0training/Adam/gradients/dense_4/Softmax_grad/subdense_4/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*"
_class
loc:@dense_4/Softmax
Ű
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Softmax_grad/mul_1*
_output_shapes
:*
data_formatNHWC*
T0*"
_class
loc:@dense_4/BiasAdd

2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Softmax_grad/mul_1dense_3/kernel/read*
transpose_b(*!
_class
loc:@dense_4/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
transpose_a( 
ó
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Relu2training/Adam/gradients/dense_4/Softmax_grad/mul_1*!
_class
loc:@dense_4/MatMul*
_output_shapes
:	*
transpose_b( *
transpose_a(*
T0
Ô
2training/Adam/gradients/dense_3/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_4/MatMul_grad/MatMuldense_3/Relu*
_class
loc:@dense_3/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
_output_shapes	
:*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
T0

2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Relu_grad/ReluGraddense_2/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a( 
ô
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Relu_grad/ReluGrad* 
_output_shapes
:
*!
_class
loc:@dense_3/MatMul*
transpose_a(*
transpose_b( *
T0
Ô
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@dense_2/Relu*
T0
Ü
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:*"
_class
loc:@dense_2/BiasAdd

2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_1/kernel/read*
transpose_a( *!
_class
loc:@dense_2/MatMul*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(
ň
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/Relu2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( *
transpose_a(* 
_output_shapes
:
*
T0*!
_class
loc:@dense_2/MatMul
Î
0training/Adam/gradients/dense/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_2/MatMul_grad/MatMul
dense/Relu*
_class
loc:@dense/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ö
6training/Adam/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0training/Adam/gradients/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:* 
_class
loc:@dense/BiasAdd*
T0
ř
0training/Adam/gradients/dense/MatMul_grad/MatMulMatMul0training/Adam/gradients/dense/Relu_grad/ReluGraddense/kernel/read*
transpose_b(*
_class
loc:@dense/MatMul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
transpose_a( 
î
2training/Adam/gradients/dense/MatMul_grad/MatMul_1MatMuldense_1_input0training/Adam/gradients/dense/Relu_grad/ReluGrad*
_class
loc:@dense/MatMul*
transpose_a(*
T0*
_output_shapes
:	1*
transpose_b( 
_
training/Adam/AssignAdd/valueConst*
dtype0	*
_output_shapes
: *
value	B	 R
Ź
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *"
_class
loc:@Adam/iterations*
T0	*
_output_shapes
: 
`
training/Adam/CastCastAdam/iterations/read*
_output_shapes
: *

DstT0*

SrcT0	
X
training/Adam/add/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
_output_shapes
: *
T0
X
training/Adam/sub/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
X
training/Adam/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *  
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
T0*
_output_shapes
: 
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
T0*
_output_shapes
: 
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
_output_shapes
: *
T0
l
training/Adam/Const_2Const*
valueB	1*    *
_output_shapes
:	1*
dtype0

training/Adam/Variable
VariableV2*
shared_name *
shape:	1*
	container *
dtype0*
_output_shapes
:	1
Ô
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/Const_2*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	1*)
_class
loc:@training/Adam/Variable

training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
T0*
_output_shapes
:	1
d
training/Adam/Const_3Const*
_output_shapes	
:*
valueB*    *
dtype0

training/Adam/Variable_1
VariableV2*
shared_name *
shape:*
	container *
dtype0*
_output_shapes	
:
Ö
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/Const_3*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_1*
T0*
validate_shape(*
use_locking(

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes	
:*
T0
n
training/Adam/Const_4Const*
valueB
*    * 
_output_shapes
:
*
dtype0

training/Adam/Variable_2
VariableV2* 
_output_shapes
:
*
shape:
*
shared_name *
dtype0*
	container 
Ű
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/Const_4*
use_locking(* 
_output_shapes
:
*
validate_shape(*+
_class!
loc:@training/Adam/Variable_2*
T0

training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
T0* 
_output_shapes
:

d
training/Adam/Const_5Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_3
VariableV2*
shape:*
	container *
shared_name *
_output_shapes	
:*
dtype0
Ö
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/Const_5*+
_class!
loc:@training/Adam/Variable_3*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_3
n
training/Adam/Const_6Const* 
_output_shapes
:
*
valueB
*    *
dtype0

training/Adam/Variable_4
VariableV2*
shape:
*
	container *
dtype0* 
_output_shapes
:
*
shared_name 
Ű
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/Const_6*
use_locking(*
T0*
validate_shape(* 
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_4

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4* 
_output_shapes
:
*
T0
d
training/Adam/Const_7Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_5
VariableV2*
	container *
dtype0*
_output_shapes	
:*
shared_name *
shape:
Ö
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/Const_7*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_5

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_5*
T0
l
training/Adam/Const_8Const*
valueB	*    *
dtype0*
_output_shapes
:	

training/Adam/Variable_6
VariableV2*
shared_name *
dtype0*
shape:	*
	container *
_output_shapes
:	
Ú
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/Const_8*
validate_shape(*
T0*+
_class!
loc:@training/Adam/Variable_6*
use_locking(*
_output_shapes
:	

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
_output_shapes
:	*
T0*+
_class!
loc:@training/Adam/Variable_6
b
training/Adam/Const_9Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_7
VariableV2*
_output_shapes
:*
shared_name *
	container *
dtype0*
shape:
Ő
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/Const_9*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
_output_shapes
:*+
_class!
loc:@training/Adam/Variable_7*
T0
m
training/Adam/Const_10Const*
dtype0*
valueB	1*    *
_output_shapes
:	1

training/Adam/Variable_8
VariableV2*
dtype0*
_output_shapes
:	1*
shape:	1*
shared_name *
	container 
Ű
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/Const_10*
_output_shapes
:	1*
use_locking(*+
_class!
loc:@training/Adam/Variable_8*
T0*
validate_shape(

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
_output_shapes
:	1*+
_class!
loc:@training/Adam/Variable_8*
T0
e
training/Adam/Const_11Const*
valueB*    *
_output_shapes	
:*
dtype0

training/Adam/Variable_9
VariableV2*
shared_name *
_output_shapes	
:*
dtype0*
	container *
shape:
×
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/Const_11*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes	
:*
validate_shape(

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_9*
T0
o
training/Adam/Const_12Const*
valueB
*    * 
_output_shapes
:
*
dtype0

training/Adam/Variable_10
VariableV2*
	container *
dtype0*
shape:
* 
_output_shapes
:
*
shared_name 
ß
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/Const_12*,
_class"
 loc:@training/Adam/Variable_10*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0

training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10* 
_output_shapes
:
*
T0*,
_class"
 loc:@training/Adam/Variable_10
e
training/Adam/Const_13Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_11
VariableV2*
	container *
_output_shapes	
:*
shape:*
shared_name *
dtype0
Ú
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/Const_13*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*,
_class"
 loc:@training/Adam/Variable_11

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes	
:
o
training/Adam/Const_14Const* 
_output_shapes
:
*
valueB
*    *
dtype0

training/Adam/Variable_12
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *
	container *
shape:

ß
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/Const_14*
T0*,
_class"
 loc:@training/Adam/Variable_12* 
_output_shapes
:
*
validate_shape(*
use_locking(

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12* 
_output_shapes
:
*
T0
e
training/Adam/Const_15Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_13
VariableV2*
_output_shapes	
:*
dtype0*
shape:*
shared_name *
	container 
Ú
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/Const_15*
use_locking(*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes	
:*
T0*
validate_shape(

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
T0*
_output_shapes	
:
m
training/Adam/Const_16Const*
dtype0*
_output_shapes
:	*
valueB	*    

training/Adam/Variable_14
VariableV2*
shape:	*
shared_name *
dtype0*
	container *
_output_shapes
:	
Ţ
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/Const_16*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0*,
_class"
 loc:@training/Adam/Variable_14

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
_output_shapes
:	*,
_class"
 loc:@training/Adam/Variable_14*
T0
c
training/Adam/Const_17Const*
dtype0*
valueB*    *
_output_shapes
:

training/Adam/Variable_15
VariableV2*
shared_name *
shape:*
	container *
dtype0*
_output_shapes
:
Ů
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/Const_17*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_15

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
T0*
_output_shapes
:
s
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
_output_shapes
:	1*
T0
Z
training/Adam/sub_2/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_2Multraining/Adam/sub_22training/Adam/gradients/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	1
n
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
_output_shapes
:	1*
T0
u
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
_output_shapes
:	1*
T0
Z
training/Adam/sub_3/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 
|
training/Adam/SquareSquare2training/Adam/gradients/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	1
o
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
T0*
_output_shapes
:	1
n
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes
:	1*
T0
l
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
_output_shapes
:	1*
T0
[
training/Adam/Const_18Const*
dtype0*
valueB
 *    *
_output_shapes
: 
[
training/Adam/Const_19Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_19*
T0*
_output_shapes
:	1

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_18*
T0*
_output_shapes
:	1
e
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
_output_shapes
:	1*
T0
Z
training/Adam/add_3/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
q
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
_output_shapes
:	1*
T0
v
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*
_output_shapes
:	1
p
training/Adam/sub_4Subdense/kernel/readtraining/Adam/truediv_1*
T0*
_output_shapes
:	1
É
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
use_locking(*)
_class
loc:@training/Adam/Variable*
_output_shapes
:	1*
validate_shape(*
T0
Ď
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
T0*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
_output_shapes
:	1*
use_locking(
ˇ
training/Adam/Assign_2Assigndense/kerneltraining/Adam/sub_4*
validate_shape(*
_output_shapes
:	1*
_class
loc:@dense/kernel*
use_locking(*
T0
q
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes	
:
Z
training/Adam/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_7Multraining/Adam/sub_56training/Adam/gradients/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
j
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes	
:
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
_output_shapes	
:*
T0
Z
training/Adam/sub_6/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
T0*
_output_shapes
: 
~
training/Adam/Square_1Square6training/Adam/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes	
:*
T0
j
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes	
:*
T0
i
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
_output_shapes	
:*
T0
[
training/Adam/Const_20Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_21Const*
_output_shapes
: *
dtype0*
valueB
 *  

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_21*
_output_shapes	
:*
T0

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_20*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes	
:*
T0
Z
training/Adam/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
m
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
_output_shapes	
:*
T0
s
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes	
:
j
training/Adam/sub_7Subdense/bias/readtraining/Adam/truediv_2*
_output_shapes	
:*
T0
Ë
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_1
Ë
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*
use_locking(*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(
Ż
training/Adam/Assign_5Assign
dense/biastraining/Adam/sub_7*
_class
loc:@dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
w
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read* 
_output_shapes
:
*
T0
Z
training/Adam/sub_8/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/dense_2/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
q
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12*
T0* 
_output_shapes
:

x
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0* 
_output_shapes
:

Z
training/Adam/sub_9/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
s
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2* 
_output_shapes
:
*
T0
q
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0* 
_output_shapes
:

n
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7* 
_output_shapes
:
*
T0
[
training/Adam/Const_22Const*
_output_shapes
: *
dtype0*
valueB
 *    
[
training/Adam/Const_23Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_23*
T0* 
_output_shapes
:


training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_22*
T0* 
_output_shapes
:

f
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0* 
_output_shapes
:

Z
training/Adam/add_9/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
r
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y*
T0* 
_output_shapes
:

x
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9* 
_output_shapes
:
*
T0
t
training/Adam/sub_10Subdense_1/kernel/readtraining/Adam/truediv_3*
T0* 
_output_shapes
:

Đ
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
T0*
validate_shape(*+
_class!
loc:@training/Adam/Variable_2* 
_output_shapes
:
*
use_locking(
Ň
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*,
_class"
 loc:@training/Adam/Variable_10*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(
˝
training/Adam/Assign_8Assigndense_1/kerneltraining/Adam/sub_10*!
_class
loc:@dense_1/kernel*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(
r
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes	
:
[
training/Adam/sub_11/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_17Multraining/Adam/sub_118training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
T0*
_output_shapes	
:
s
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
_output_shapes	
:*
T0
[
training/Adam/sub_12/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes	
:
m
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
_output_shapes	
:*
T0
j
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
_output_shapes	
:*
T0
[
training/Adam/Const_24Const*
valueB
 *    *
_output_shapes
: *
dtype0
[
training/Adam/Const_25Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_25*
_output_shapes	
:*
T0

training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_24*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes	
:
[
training/Adam/add_12/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
o
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes	
:
m
training/Adam/sub_13Subdense_1/bias/readtraining/Adam/truediv_4*
T0*
_output_shapes	
:
Ě
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
validate_shape(*
use_locking(*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_3*
T0
Ď
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
validate_shape(*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes	
:
ľ
training/Adam/Assign_11Assigndense_1/biastraining/Adam/sub_13*
use_locking(*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes	
:*
T0
w
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0* 
_output_shapes
:

[
training/Adam/sub_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

r
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0* 
_output_shapes
:

x
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read* 
_output_shapes
:
*
T0
[
training/Adam/sub_15/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_4Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

t
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0* 
_output_shapes
:

r
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24* 
_output_shapes
:
*
T0
o
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13*
T0* 
_output_shapes
:

[
training/Adam/Const_26Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_27Const*
_output_shapes
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_27*
T0* 
_output_shapes
:


training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_26* 
_output_shapes
:
*
T0
f
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5*
T0* 
_output_shapes
:

[
training/Adam/add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
t
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y*
T0* 
_output_shapes
:

y
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15* 
_output_shapes
:
*
T0
t
training/Adam/sub_16Subdense_2/kernel/readtraining/Adam/truediv_5*
T0* 
_output_shapes
:

Ň
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_4* 
_output_shapes
:
*
T0
Ô
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_12
ž
training/Adam/Assign_14Assigndense_2/kerneltraining/Adam/sub_16*
use_locking(*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
*
validate_shape(*
T0
r
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
_output_shapes	
:*
T0
[
training/Adam/sub_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
_output_shapes	
:*
T0
s
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
_output_shapes	
:*
T0
[
training/Adam/sub_18/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
_output_shapes	
:*
T0
m
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes	
:
j
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
_output_shapes	
:*
T0
[
training/Adam/Const_28Const*
valueB
 *    *
_output_shapes
: *
dtype0
[
training/Adam/Const_29Const*
valueB
 *  *
_output_shapes
: *
dtype0

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_29*
_output_shapes	
:*
T0

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_28*
T0*
_output_shapes	
:
a
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes	
:
[
training/Adam/add_18/yConst*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
o
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
_output_shapes	
:*
T0
m
training/Adam/sub_19Subdense_2/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes	
:
Í
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
validate_shape(*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_5*
use_locking(
Ď
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
_output_shapes	
:*
use_locking(*,
_class"
 loc:@training/Adam/Variable_13*
T0*
validate_shape(
ľ
training/Adam/Assign_17Assigndense_2/biastraining/Adam/sub_19*
_output_shapes	
:*
T0*
validate_shape(*
_class
loc:@dense_2/bias*
use_locking(
v
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes
:	
[
training/Adam/sub_20/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
q
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes
:	
w
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
_output_shapes
:	*
T0
[
training/Adam/sub_21/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
s
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
_output_shapes
:	*
T0
q
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
_output_shapes
:	*
T0
n
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes
:	
[
training/Adam/Const_30Const*
_output_shapes
: *
dtype0*
valueB
 *    
[
training/Adam/Const_31Const*
_output_shapes
: *
dtype0*
valueB
 *  

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_31*
T0*
_output_shapes
:	

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_30*
T0*
_output_shapes
:	
e
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes
:	*
T0
[
training/Adam/add_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
s
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes
:	
x
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
T0*
_output_shapes
:	
s
training/Adam/sub_22Subdense_3/kernel/readtraining/Adam/truediv_7*
T0*
_output_shapes
:	
Ń
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
_output_shapes
:	*
use_locking(*
validate_shape(*+
_class!
loc:@training/Adam/Variable_6*
T0
Ó
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(
˝
training/Adam/Assign_20Assigndense_3/kerneltraining/Adam/sub_22*
_output_shapes
:	*!
_class
loc:@dense_3/kernel*
validate_shape(*
use_locking(*
T0
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
:
[
training/Adam/sub_23/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_37Multraining/Adam/sub_238training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
_output_shapes
:*
T0
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
:
[
training/Adam/sub_24/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_7Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
T0*
_output_shapes
:
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
_output_shapes
:*
T0
[
training/Adam/Const_32Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_33Const*
dtype0*
valueB
 *  *
_output_shapes
: 

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_33*
_output_shapes
:*
T0

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_32*
_output_shapes
:*
T0
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
T0*
_output_shapes
:
[
training/Adam/add_24/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
T0*
_output_shapes
:
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
_output_shapes
:*
T0
l
training/Adam/sub_25Subdense_3/bias/readtraining/Adam/truediv_8*
T0*
_output_shapes
:
Ě
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
_output_shapes
:*+
_class!
loc:@training/Adam/Variable_7*
use_locking(*
T0*
validate_shape(
Î
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15*
T0*
validate_shape(*
_output_shapes
:
´
training/Adam/Assign_23Assigndense_3/biastraining/Adam/sub_25*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*
_class
loc:@dense_3/bias
ˇ
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/Adam/AssignAdd^training/Adam/Assign^training/Adam/Assign_1^training/Adam/Assign_2^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23
0

group_depsNoOp	^loss/mul^metrics/acc/Mean

IsVariableInitializedIsVariableInitializeddense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitialized
dense/bias*
_output_shapes
: *
dtype0*
_class
loc:@dense/bias

IsVariableInitialized_2IsVariableInitializeddense_1/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel

IsVariableInitialized_3IsVariableInitializeddense_1/bias*
_output_shapes
: *
dtype0*
_class
loc:@dense_1/bias

IsVariableInitialized_4IsVariableInitializeddense_2/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_2/kernel

IsVariableInitialized_5IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_6IsVariableInitializeddense_3/kernel*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0

IsVariableInitialized_7IsVariableInitializeddense_3/bias*
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
_output_shapes
: *
dtype0	
z
IsVariableInitialized_9IsVariableInitializedAdam/lr*
dtype0*
_output_shapes
: *
_class
loc:@Adam/lr

IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
dtype0*
_class
loc:@Adam/beta_1*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedAdam/beta_2*
_output_shapes
: *
dtype0*
_class
loc:@Adam/beta_2

IsVariableInitialized_12IsVariableInitialized
Adam/decay*
_class
loc:@Adam/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable*
dtype0*
_output_shapes
: *)
_class
loc:@training/Adam/Variable

IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_1*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_1

IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_2*
_output_shapes
: *
dtype0*+
_class!
loc:@training/Adam/Variable_2

IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_3*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_3

IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*
dtype0*+
_class!
loc:@training/Adam/Variable_4*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*+
_class!
loc:@training/Adam/Variable_5*
dtype0*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_6*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_6

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_7*
_output_shapes
: *
dtype0*+
_class!
loc:@training/Adam/Variable_7

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*
dtype0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
dtype0*
_output_shapes
: 

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_10*
_output_shapes
: *
dtype0*,
_class"
 loc:@training/Adam/Variable_10

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
: *
dtype0

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_12*
_output_shapes
: *
dtype0*,
_class"
 loc:@training/Adam/Variable_12

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*
_output_shapes
: *
dtype0*,
_class"
 loc:@training/Adam/Variable_13

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_14*
_output_shapes
: *
dtype0*,
_class"
 loc:@training/Adam/Variable_14

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_15*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
: *
dtype0
Ě
initNoOp^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign ^training/Adam/Variable_2/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign"ŕňŐű     ^nŠ	5^yň0Ń×AJî­
!ó 
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
8
FloorMod
x"T
y"T
z"T"
Ttype:	
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	

C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2

#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
1
Square
x"T
y"T"
Ttype:

2	
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.5.02v1.5.0-0-g37aa430d84žÚ
p
dense_1_inputPlaceholder*
shape:˙˙˙˙˙˙˙˙˙1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
dtype0

-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
:*
valueB"1      

+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *<ž*
_output_shapes
: *
_class
loc:@dense/kernel

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *<>*
_output_shapes
: *
_class
loc:@dense/kernel*
dtype0
ć
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*

seed *
_output_shapes
:	1*
_class
loc:@dense/kernel*
dtype0*
T0*
seed2 
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
á
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel*
T0*
_output_shapes
:	1
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	1*
_class
loc:@dense/kernel*
T0
Ł
dense/kernel
VariableV2*
shared_name *
	container *
shape:	1*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
:	1
Č
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	1*
use_locking(
v
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel*
_output_shapes
:	1*
T0

dense/bias/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
_class
loc:@dense/bias*
dtype0


dense/bias
VariableV2*
shared_name *
_output_shapes	
:*
dtype0*
	container *
_class
loc:@dense/bias*
shape:
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
_class
loc:@dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
_output_shapes	
:*
T0

dense/MatMulMatMuldense_1_inputdense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( *
T0

dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

dense/ReluReludense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*!
_class
loc:@dense_1/kernel

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *   ž*
_output_shapes
: *
dtype0*!
_class
loc:@dense_1/kernel

-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *   >*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
í
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*

seed *!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
dtype0*
seed2 
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *!
_class
loc:@dense_1/kernel
ę
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel* 
_output_shapes
:

Ü
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*
T0*!
_class
loc:@dense_1/kernel
Š
dense_1/kernel
VariableV2*
	container *
dtype0*
shape:
* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel*
shared_name 
Ń
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
T0* 
_output_shapes
:
*
use_locking(
}
dense_1/kernel/readIdentitydense_1/kernel* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel*
T0

dense_1/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*
_class
loc:@dense_1/bias

dense_1/bias
VariableV2*
dtype0*
	container *
_class
loc:@dense_1/bias*
shared_name *
shape:*
_output_shapes	
:
ť
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(*
_class
loc:@dense_1/bias
r
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
T0*
_output_shapes	
:

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
transpose_a( *
T0

dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
X
dense_2/ReluReludense_2/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_2/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:

-dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *óľ˝*!
_class
loc:@dense_2/kernel

-dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *óľ=*!
_class
loc:@dense_2/kernel
í
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*

seed * 
_output_shapes
:
*
seed2 *
T0*
dtype0*!
_class
loc:@dense_2/kernel
Ö
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
ę
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
*
T0
Ü
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*!
_class
loc:@dense_2/kernel
Š
dense_2/kernel
VariableV2*
shared_name *!
_class
loc:@dense_2/kernel*
shape:
*
dtype0*
	container * 
_output_shapes
:

Ń
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
*
use_locking(*
T0
}
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:


dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
_class
loc:@dense_2/bias*
valueB*    

dense_2/bias
VariableV2*
dtype0*
shared_name *
	container *
_output_shapes	
:*
_class
loc:@dense_2/bias*
shape:
ť
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*
_class
loc:@dense_2/bias
r
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes	
:

dense_3/MatMulMatMuldense_2/Reludense_2/kernel/read*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 

dense_3/BiasAddBiasAdddense_3/MatMuldense_2/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
X
dense_3/ReluReludense_3/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *!
_class
loc:@dense_3/kernel

-dense_3/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *żđÚ˝*!
_class
loc:@dense_3/kernel*
dtype0

-dense_3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *żđÚ=*!
_class
loc:@dense_3/kernel*
dtype0
ě
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*

seed *
dtype0*
T0*
seed2 *!
_class
loc:@dense_3/kernel*
_output_shapes
:	
Ö
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
T0
é
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*!
_class
loc:@dense_3/kernel*
T0
Ű
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	
§
dense_3/kernel
VariableV2*!
_class
loc:@dense_3/kernel*
_output_shapes
:	*
	container *
shape:	*
shared_name *
dtype0
Đ
dense_3/kernel/AssignAssigndense_3/kernel)dense_3/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0*!
_class
loc:@dense_3/kernel
|
dense_3/kernel/readIdentitydense_3/kernel*
T0*
_output_shapes
:	*!
_class
loc:@dense_3/kernel

dense_3/bias/Initializer/zerosConst*
_output_shapes
:*
_class
loc:@dense_3/bias*
dtype0*
valueB*    

dense_3/bias
VariableV2*
_output_shapes
:*
dtype0*
shape:*
	container *
shared_name *
_class
loc:@dense_3/bias
ş
dense_3/bias/AssignAssigndense_3/biasdense_3/bias/Initializer/zeros*
_class
loc:@dense_3/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:*
_class
loc:@dense_3/bias*
T0

dense_4/MatMulMatMuldense_3/Reludense_3/kernel/read*
T0*
transpose_b( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_4/BiasAddBiasAdddense_4/MatMuldense_3/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
]
dense_4/SoftmaxSoftmaxdense_4/BiasAdd*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
_
Adam/iterations/initial_valueConst*
_output_shapes
: *
value	B	 R *
dtype0	
s
Adam/iterations
VariableV2*
shared_name *
dtype0	*
shape: *
_output_shapes
: *
	container 
ž
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
_output_shapes
: *
T0	*
use_locking(*
validate_shape(*"
_class
loc:@Adam/iterations
v
Adam/iterations/readIdentityAdam/iterations*
_output_shapes
: *
T0	*"
_class
loc:@Adam/iterations
Z
Adam/lr/initial_valueConst*
valueB
 *ˇŃ8*
_output_shapes
: *
dtype0
k
Adam/lr
VariableV2*
shape: *
shared_name *
_output_shapes
: *
	container *
dtype0

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_class
loc:@Adam/lr*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
^
Adam/lr/readIdentityAdam/lr*
_class
loc:@Adam/lr*
_output_shapes
: *
T0
^
Adam/beta_1/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
dtype0
o
Adam/beta_1
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
Ž
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
validate_shape(*
_class
loc:@Adam/beta_1*
_output_shapes
: *
use_locking(*
T0
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_output_shapes
: *
_class
loc:@Adam/beta_1
^
Adam/beta_2/initial_valueConst*
valueB
 *wž?*
dtype0*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
	container *
_output_shapes
: *
shape: *
dtype0*
shared_name 
Ž
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_class
loc:@Adam/beta_2*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
j
Adam/beta_2/readIdentityAdam/beta_2*
_output_shapes
: *
_class
loc:@Adam/beta_2*
T0
]
Adam/decay/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n

Adam/decay
VariableV2*
dtype0*
_output_shapes
: *
	container *
shared_name *
shape: 
Ş
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
validate_shape(*
T0*
use_locking(*
_class
loc:@Adam/decay*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
T0*
_class
loc:@Adam/decay*
_output_shapes
: 

dense_4_targetPlaceholder*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
q
dense_4_sample_weightsPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
loss/dense_4_loss/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *żÖ3
\
loss/dense_4_loss/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
loss/dense_4_loss/subSubloss/dense_4_loss/sub/xloss/dense_4_loss/Const*
_output_shapes
: *
T0

'loss/dense_4_loss/clip_by_value/MinimumMinimumdense_4/Softmaxloss/dense_4_loss/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

loss/dense_4_loss/clip_by_valueMaximum'loss/dense_4_loss/clip_by_value/Minimumloss/dense_4_loss/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
o
loss/dense_4_loss/LogLogloss/dense_4_loss/clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
loss/dense_4_loss/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

loss/dense_4_loss/ReshapeReshapedense_4_targetloss/dense_4_loss/Reshape/shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
v
loss/dense_4_loss/CastCastloss/dense_4_loss/Reshape*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0	
r
!loss/dense_4_loss/Reshape_1/shapeConst*
valueB"˙˙˙˙   *
_output_shapes
:*
dtype0
 
loss/dense_4_loss/Reshape_1Reshapeloss/dense_4_loss/Log!loss/dense_4_loss/Reshape_1/shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

;loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/dense_4_loss/Cast*
out_type0*
T0	*
_output_shapes
:

Yloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss/dense_4_loss/Reshape_1loss/dense_4_loss/Cast*
T0*
Tlabels0	*6
_output_shapes$
":˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
k
(loss/dense_4_loss/Mean/reduction_indicesConst*
_output_shapes
: *
valueB *
dtype0
î
loss/dense_4_loss/MeanMeanYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(loss/dense_4_loss/Mean/reduction_indices*
	keep_dims( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
z
loss/dense_4_loss/mulMulloss/dense_4_loss/Meandense_4_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
loss/dense_4_loss/NotEqual/yConst*
valueB
 *    *
_output_shapes
: *
dtype0

loss/dense_4_loss/NotEqualNotEqualdense_4_sample_weightsloss/dense_4_loss/NotEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
loss/dense_4_loss/Cast_1Castloss/dense_4_loss/NotEqual*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0

c
loss/dense_4_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

loss/dense_4_loss/Mean_1Meanloss/dense_4_loss/Cast_1loss/dense_4_loss/Const_1*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0

loss/dense_4_loss/truedivRealDivloss/dense_4_loss/mulloss/dense_4_loss/Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
loss/dense_4_loss/Const_2Const*
valueB: *
_output_shapes
:*
dtype0

loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
O

loss/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
V
loss/mulMul
loss/mul/xloss/dense_4_loss/Mean_2*
_output_shapes
: *
T0
l
!metrics/acc/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

metrics/acc/MaxMaxdense_4_target!metrics/acc/Max/reduction_indices*
	keep_dims( *

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
g
metrics/acc/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

metrics/acc/ArgMaxArgMaxdense_4/Softmaxmetrics/acc/ArgMax/dimension*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
i
metrics/acc/CastCastmetrics/acc/ArgMax*

SrcT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
k
metrics/acc/EqualEqualmetrics/acc/Maxmetrics/acc/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
j
metrics/acc/Cast_1Castmetrics/acc/Equal*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0*

SrcT0

[
metrics/acc/ConstConst*
valueB: *
_output_shapes
:*
dtype0
}
metrics/acc/MeanMeanmetrics/acc/Cast_1metrics/acc/Const*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
}
training/Adam/gradients/ShapeConst*
valueB *
_output_shapes
: *
_class
loc:@loss/mul*
dtype0

!training/Adam/gradients/grad_ys_0Const*
valueB
 *  ?*
_class
loc:@loss/mul*
_output_shapes
: *
dtype0
¤
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
_class
loc:@loss/mul*
_output_shapes
: *
T0
Ś
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_2*
T0*
_output_shapes
: *
_class
loc:@loss/mul

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_class
loc:@loss/mul*
T0*
_output_shapes
: 
ş
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0*+
_class!
loc:@loss/dense_4_loss/Mean_2

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape/shape*
T0*
Tshape0*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
:
Á
;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ShapeShapeloss/dense_4_loss/truediv*
out_type0*
T0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_2
Ť
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0*+
_class!
loc:@loss/dense_4_loss/Mean_2
Ă
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_1Shapeloss/dense_4_loss/truediv*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
:*
T0*
out_type0
­
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_2Const*
valueB *
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_2*
dtype0
˛
;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ConstConst*+
_class!
loc:@loss/dense_4_loss/Mean_2*
valueB: *
dtype0*
_output_shapes
:
Š
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_2
´
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *+
_class!
loc:@loss/dense_4_loss/Mean_2
­
<training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const_1*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
Ž
?training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_2

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/MaximumMaximum<training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0*+
_class!
loc:@loss/dense_4_loss/Mean_2

>training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum*+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0*
_output_shapes
: 
ß
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_2*

DstT0

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Cast*+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
_output_shapes
:*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*
out_type0
Ż
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*
valueB *
dtype0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
: 
Î
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ţ
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truedivloss/dense_4_loss/Mean_1*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*,
_class"
 loc:@loss/dense_4_loss/truediv*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
­
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0*
T0
´
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0
ý
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Negloss/dense_4_loss/Mean_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv

@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1loss/dense_4_loss/Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0

:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulMul=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truediv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv
˝
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*,
_class"
 loc:@loss/dense_4_loss/truediv*

Tidx0*
	keep_dims( 
Ś
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
: *
T0*
Tshape0
¸
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean*
out_type0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:*
T0
ş
:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1Shapedense_4_sample_weights*
_output_shapes
:*(
_class
loc:@loss/dense_4_loss/mul*
out_type0*
T0
ž
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*(
_class
loc:@loss/dense_4_loss/mul
í
6training/Adam/gradients/loss/dense_4_loss/mul_grad/mulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul
Š
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/mulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 

:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0
ď
8training/Adam/gradients/loss/dense_4_loss/mul_grad/mul_1Mulloss/dense_4_loss/Mean>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul*
T0
Ż
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( *(
_class
loc:@loss/dense_4_loss/mul
Ł
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*(
_class
loc:@loss/dense_4_loss/mul*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ý
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
out_type0*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:
Ľ
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
value	B :*
dtype0
đ
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
_output_shapes
: 

7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
_output_shapes
: 
°
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*
valueB: *
dtype0*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/startConst*
dtype0*)
_class
loc:@loss/dense_4_loss/Mean*
value	B : *
_output_shapes
: 
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/deltaConst*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
_output_shapes
: *
value	B :
Ń
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*

Tidx0
Ť
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/valueConst*
_output_shapes
: *
dtype0*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :

8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean

Atraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_4_loss/Mean_grad/range7training/Adam/gradients/loss/dense_4_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill*)
_class
loc:@loss/dense_4_loss/Mean*
N*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
Ą
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*)
_class
loc:@loss/dense_4_loss/Mean

<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*)
_class
loc:@loss/dense_4_loss/Mean
Ą
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*)
_class
loc:@loss/dense_4_loss/Mean*
Tshape0*
_output_shapes
:*
T0

8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*

Tmultiples0*
T0
˙
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_2ShapeYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
out_type0
ź
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*
out_type0*
_output_shapes
:*
T0*)
_class
loc:@loss/dense_4_loss/Mean
Ž
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
Ą
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*
T0*

Tidx0*
	keep_dims( *)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
°
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0*
valueB: 
Ľ
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
	keep_dims( *

Tidx0*
T0
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
dtype0

=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/y*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *
T0

>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
_output_shapes
: 
Ű
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*

DstT0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *

SrcT0

;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*)
_class
loc:@loss/dense_4_loss/Mean*
T0
Ź
"training/Adam/gradients/zeros_like	ZerosLike[loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0
Î
training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient[loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*´
message¨ĽCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
ż
training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙

training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivtraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*

Tdim0*
T0
Ž
ztraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMultraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimstraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ă
>training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/ShapeShapeloss/dense_4_loss/Log*.
_class$
" loc:@loss/dense_4_loss/Reshape_1*
_output_shapes
:*
T0*
out_type0
÷
@training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/ReshapeReshapeztraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul>training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Shape*
T0*.
_class$
" loc:@loss/dense_4_loss/Reshape_1*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

=training/Adam/gradients/loss/dense_4_loss/Log_grad/Reciprocal
Reciprocalloss/dense_4_loss/clip_by_valueA^training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/Log*
T0

6training/Adam/gradients/loss/dense_4_loss/Log_grad/mulMul@training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Reshape=training/Adam/gradients/loss/dense_4_loss/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*(
_class
loc:@loss/dense_4_loss/Log
Ý
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ShapeShape'loss/dense_4_loss/clip_by_value/Minimum*
out_type0*
T0*
_output_shapes
:*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
ť
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
î
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_2Shape6training/Adam/gradients/loss/dense_4_loss/Log_grad/mul*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
out_type0*
T0*
_output_shapes
:
Á
Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
dtype0*
_output_shapes
: 
Ŕ
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zerosFillDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_2Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0

Itraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_4_loss/clip_by_value/Minimumloss/dense_4_loss/Const*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ć
Rtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ShapeDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1*
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ú
Ctraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SelectSelectItraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqual6training/Adam/gradients/loss/dense_4_loss/Log_grad/mulBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ü
Etraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Select_1SelectItraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqualBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros6training/Adam/gradients/loss/dense_4_loss/Log_grad/mul*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ô
@training/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SumSumCtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SelectRtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
É
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0
Ú
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Sum_1SumEtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Select_1Ttraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*

Tidx0*
T0
ž
Ftraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1*
_output_shapes
: *
Tshape0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0
Ő
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ShapeShapedense_4/Softmax*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:*
T0
Ë
Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
_output_shapes
: *
valueB 

Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_2ShapeDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape*
out_type0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0*
_output_shapes
:
Ń
Ptraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
_output_shapes
: *
dtype0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
valueB
 *    
ŕ
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros/Const*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ń
Ntraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualdense_4/Softmaxloss/dense_4_loss/sub*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

Ztraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0
Ľ
Ktraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SelectSelectNtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqualDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ReshapeJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
Mtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Select_1SelectNtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqualJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zerosDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ô
Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
	keep_dims( 
é
Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
ú
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
Ţ
Ntraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
Tshape0*
_output_shapes
: 
ě
0training/Adam/gradients/dense_4/Softmax_grad/mulMulLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshapedense_4/Softmax*
T0*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
Btraining/Adam/gradients/dense_4/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0*"
_class
loc:@dense_4/Softmax

0training/Adam/gradients/dense_4/Softmax_grad/SumSum0training/Adam/gradients/dense_4/Softmax_grad/mulBtraining/Adam/gradients/dense_4/Softmax_grad/Sum/reduction_indices*
	keep_dims( *"
_class
loc:@dense_4/Softmax*
T0*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
:training/Adam/gradients/dense_4/Softmax_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   *"
_class
loc:@dense_4/Softmax

4training/Adam/gradients/dense_4/Softmax_grad/ReshapeReshape0training/Adam/gradients/dense_4/Softmax_grad/Sum:training/Adam/gradients/dense_4/Softmax_grad/Reshape/shape*
Tshape0*"
_class
loc:@dense_4/Softmax*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

0training/Adam/gradients/dense_4/Softmax_grad/subSubLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshape4training/Adam/gradients/dense_4/Softmax_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*"
_class
loc:@dense_4/Softmax
Ň
2training/Adam/gradients/dense_4/Softmax_grad/mul_1Mul0training/Adam/gradients/dense_4/Softmax_grad/subdense_4/Softmax*
T0*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Softmax_grad/mul_1*
data_formatNHWC*
_output_shapes
:*
T0*"
_class
loc:@dense_4/BiasAdd

2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Softmax_grad/mul_1dense_3/kernel/read*!
_class
loc:@dense_4/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0*
transpose_b(
ó
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Relu2training/Adam/gradients/dense_4/Softmax_grad/mul_1*
transpose_b( *
_output_shapes
:	*
transpose_a(*
T0*!
_class
loc:@dense_4/MatMul
Ô
2training/Adam/gradients/dense_3/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_4/MatMul_grad/MatMuldense_3/Relu*
_class
loc:@dense_3/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ü
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*"
_class
loc:@dense_3/BiasAdd*
data_formatNHWC*
T0*
_output_shapes	
:

2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Relu_grad/ReluGraddense_2/kernel/read*!
_class
loc:@dense_3/MatMul*
T0*
transpose_b(*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ô
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
transpose_a(*!
_class
loc:@dense_3/MatMul*
transpose_b( *
T0* 
_output_shapes
:

Ô
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@dense_2/Relu*
T0
Ü
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*
_output_shapes	
:*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC

2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_1/kernel/read*!
_class
loc:@dense_2/MatMul*
transpose_b(*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ň
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/Relu2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( * 
_output_shapes
:
*!
_class
loc:@dense_2/MatMul*
transpose_a(*
T0
Î
0training/Adam/gradients/dense/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_2/MatMul_grad/MatMul
dense/Relu*
_class
loc:@dense/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ö
6training/Adam/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0training/Adam/gradients/dense/Relu_grad/ReluGrad*
data_formatNHWC* 
_class
loc:@dense/BiasAdd*
_output_shapes	
:*
T0
ř
0training/Adam/gradients/dense/MatMul_grad/MatMulMatMul0training/Adam/gradients/dense/Relu_grad/ReluGraddense/kernel/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
transpose_b(*
transpose_a( *
_class
loc:@dense/MatMul
î
2training/Adam/gradients/dense/MatMul_grad/MatMul_1MatMuldense_1_input0training/Adam/gradients/dense/Relu_grad/ReluGrad*
T0*
transpose_a(*
transpose_b( *
_output_shapes
:	1*
_class
loc:@dense/MatMul
_
training/Adam/AssignAdd/valueConst*
value	B	 R*
dtype0	*
_output_shapes
: 
Ź
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *"
_class
loc:@Adam/iterations*
_output_shapes
: *
T0	
`
training/Adam/CastCastAdam/iterations/read*
_output_shapes
: *

DstT0*

SrcT0	
X
training/Adam/add/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
_output_shapes
: *
T0
X
training/Adam/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
a
training/Adam/subSubtraining/Adam/sub/xtraining/Adam/Pow*
_output_shapes
: *
T0
X
training/Adam/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
Z
training/Adam/Const_1Const*
_output_shapes
: *
valueB
 *  *
dtype0
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
T0*
_output_shapes
: 

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
T0*
_output_shapes
: 
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
T0*
_output_shapes
: 
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
T0*
_output_shapes
: 
Z
training/Adam/sub_1/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
g
training/Adam/sub_1Subtraining/Adam/sub_1/xtraining/Adam/Pow_1*
_output_shapes
: *
T0
j
training/Adam/truedivRealDivtraining/Adam/Sqrttraining/Adam/sub_1*
_output_shapes
: *
T0
^
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
l
training/Adam/Const_2Const*
dtype0*
_output_shapes
:	1*
valueB	1*    

training/Adam/Variable
VariableV2*
dtype0*
	container *
_output_shapes
:	1*
shape:	1*
shared_name 
Ô
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/Const_2*
validate_shape(*
T0*)
_class
loc:@training/Adam/Variable*
use_locking(*
_output_shapes
:	1

training/Adam/Variable/readIdentitytraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
_output_shapes
:	1*
T0
d
training/Adam/Const_3Const*
dtype0*
valueB*    *
_output_shapes	
:

training/Adam/Variable_1
VariableV2*
	container *
dtype0*
shape:*
shared_name *
_output_shapes	
:
Ö
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/Const_3*
validate_shape(*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_1*
use_locking(

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_1
n
training/Adam/Const_4Const*
dtype0* 
_output_shapes
:
*
valueB
*    

training/Adam/Variable_2
VariableV2*
dtype0*
shared_name * 
_output_shapes
:
*
	container *
shape:

Ű
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/Const_4* 
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
use_locking(*
T0

training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0*+
_class!
loc:@training/Adam/Variable_2* 
_output_shapes
:

d
training/Adam/Const_5Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_3
VariableV2*
	container *
dtype0*
shape:*
shared_name *
_output_shapes	
:
Ö
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/Const_5*+
_class!
loc:@training/Adam/Variable_3*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_3*
T0
n
training/Adam/Const_6Const*
dtype0* 
_output_shapes
:
*
valueB
*    

training/Adam/Variable_4
VariableV2* 
_output_shapes
:
*
shared_name *
dtype0*
shape:
*
	container 
Ű
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/Const_6*+
_class!
loc:@training/Adam/Variable_4*
T0*
use_locking(* 
_output_shapes
:
*
validate_shape(

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4* 
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_4*
T0
d
training/Adam/Const_7Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_5
VariableV2*
shared_name *
	container *
shape:*
dtype0*
_output_shapes	
:
Ö
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/Const_7*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_5
l
training/Adam/Const_8Const*
valueB	*    *
_output_shapes
:	*
dtype0

training/Adam/Variable_6
VariableV2*
	container *
dtype0*
shared_name *
shape:	*
_output_shapes
:	
Ú
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/Const_8*
validate_shape(*
T0*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes
:	*
use_locking(

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
_output_shapes
:	*+
_class!
loc:@training/Adam/Variable_6*
T0
b
training/Adam/Const_9Const*
_output_shapes
:*
dtype0*
valueB*    

training/Adam/Variable_7
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
Ő
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/Const_9*
use_locking(*
validate_shape(*
_output_shapes
:*+
_class!
loc:@training/Adam/Variable_7*
T0

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
_output_shapes
:*
T0*+
_class!
loc:@training/Adam/Variable_7
m
training/Adam/Const_10Const*
_output_shapes
:	1*
dtype0*
valueB	1*    

training/Adam/Variable_8
VariableV2*
shared_name *
	container *
shape:	1*
dtype0*
_output_shapes
:	1
Ű
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/Const_10*
_output_shapes
:	1*
validate_shape(*
T0*+
_class!
loc:@training/Adam/Variable_8*
use_locking(

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
_output_shapes
:	1*
T0*+
_class!
loc:@training/Adam/Variable_8
e
training/Adam/Const_11Const*
valueB*    *
_output_shapes	
:*
dtype0

training/Adam/Variable_9
VariableV2*
shape:*
	container *
dtype0*
shared_name *
_output_shapes	
:
×
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/Const_11*
use_locking(*
validate_shape(*
T0*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes	
:

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_9*
T0
o
training/Adam/Const_12Const*
dtype0* 
_output_shapes
:
*
valueB
*    

training/Adam/Variable_10
VariableV2*
shape:
* 
_output_shapes
:
*
	container *
shared_name *
dtype0
ß
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/Const_12*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0*,
_class"
 loc:@training/Adam/Variable_10

training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0*,
_class"
 loc:@training/Adam/Variable_10* 
_output_shapes
:

e
training/Adam/Const_13Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_11
VariableV2*
dtype0*
	container *
shape:*
shared_name *
_output_shapes	
:
Ú
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/Const_13*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*
_output_shapes	
:*,
_class"
 loc:@training/Adam/Variable_11
o
training/Adam/Const_14Const*
valueB
*    * 
_output_shapes
:
*
dtype0

training/Adam/Variable_12
VariableV2*
shape:
*
	container * 
_output_shapes
:
*
shared_name *
dtype0
ß
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/Const_14* 
_output_shapes
:
*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_12

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0*,
_class"
 loc:@training/Adam/Variable_12* 
_output_shapes
:

e
training/Adam/Const_15Const*
_output_shapes	
:*
dtype0*
valueB*    

training/Adam/Variable_13
VariableV2*
	container *
shape:*
shared_name *
dtype0*
_output_shapes	
:
Ú
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/Const_15*
T0*,
_class"
 loc:@training/Adam/Variable_13*
use_locking(*
_output_shapes	
:*
validate_shape(

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes	
:
m
training/Adam/Const_16Const*
_output_shapes
:	*
dtype0*
valueB	*    

training/Adam/Variable_14
VariableV2*
shared_name *
shape:	*
	container *
dtype0*
_output_shapes
:	
Ţ
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/Const_16*
use_locking(*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_14*
T0*
_output_shapes
:	

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes
:	*
T0
c
training/Adam/Const_17Const*
valueB*    *
_output_shapes
:*
dtype0

training/Adam/Variable_15
VariableV2*
shape:*
_output_shapes
:*
dtype0*
shared_name *
	container 
Ů
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/Const_17*,
_class"
 loc:@training/Adam/Variable_15*
T0*
validate_shape(*
_output_shapes
:*
use_locking(

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_15
s
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
T0*
_output_shapes
:	1
Z
training/Adam/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_2Multraining/Adam/sub_22training/Adam/gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	1*
T0
n
training/Adam/add_1Addtraining/Adam/mul_1training/Adam/mul_2*
T0*
_output_shapes
:	1
u
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
_output_shapes
:	1*
T0
Z
training/Adam/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
d
training/Adam/sub_3Subtraining/Adam/sub_3/xAdam/beta_2/read*
T0*
_output_shapes
: 
|
training/Adam/SquareSquare2training/Adam/gradients/dense/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	1
o
training/Adam/mul_4Multraining/Adam/sub_3training/Adam/Square*
_output_shapes
:	1*
T0
n
training/Adam/add_2Addtraining/Adam/mul_3training/Adam/mul_4*
_output_shapes
:	1*
T0
l
training/Adam/mul_5Multraining/Adam/multraining/Adam/add_1*
_output_shapes
:	1*
T0
[
training/Adam/Const_18Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_19Const*
valueB
 *  *
_output_shapes
: *
dtype0

%training/Adam/clip_by_value_1/MinimumMinimumtraining/Adam/add_2training/Adam/Const_19*
T0*
_output_shapes
:	1

training/Adam/clip_by_value_1Maximum%training/Adam/clip_by_value_1/Minimumtraining/Adam/Const_18*
T0*
_output_shapes
:	1
e
training/Adam/Sqrt_1Sqrttraining/Adam/clip_by_value_1*
T0*
_output_shapes
:	1
Z
training/Adam/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
q
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
_output_shapes
:	1*
T0
v
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
T0*
_output_shapes
:	1
p
training/Adam/sub_4Subdense/kernel/readtraining/Adam/truediv_1*
_output_shapes
:	1*
T0
É
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*)
_class
loc:@training/Adam/Variable*
use_locking(*
validate_shape(*
_output_shapes
:	1*
T0
Ď
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
:	1*
T0*
use_locking(*
validate_shape(
ˇ
training/Adam/Assign_2Assigndense/kerneltraining/Adam/sub_4*
validate_shape(*
_output_shapes
:	1*
T0*
_class
loc:@dense/kernel*
use_locking(
q
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
T0*
_output_shapes	
:
Z
training/Adam/sub_5/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_7Multraining/Adam/sub_56training/Adam/gradients/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
j
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
_output_shapes	
:*
T0
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
_output_shapes	
:*
T0
Z
training/Adam/sub_6/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
d
training/Adam/sub_6Subtraining/Adam/sub_6/xAdam/beta_2/read*
_output_shapes
: *
T0
~
training/Adam/Square_1Square6training/Adam/gradients/dense/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
m
training/Adam/mul_9Multraining/Adam/sub_6training/Adam/Square_1*
_output_shapes	
:*
T0
j
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
_output_shapes	
:*
T0
i
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes	
:
[
training/Adam/Const_20Const*
dtype0*
valueB
 *    *
_output_shapes
: 
[
training/Adam/Const_21Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_21*
T0*
_output_shapes	
:

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_20*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_2Sqrttraining/Adam/clip_by_value_2*
_output_shapes	
:*
T0
Z
training/Adam/add_6/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
m
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes	
:
s
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
T0*
_output_shapes	
:
j
training/Adam/sub_7Subdense/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes	
:
Ë
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_1
Ë
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
Ż
training/Adam/Assign_5Assign
dense/biastraining/Adam/sub_7*
_class
loc:@dense/bias*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
w
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0* 
_output_shapes
:

Z
training/Adam/sub_8/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_12Multraining/Adam/sub_84training/Adam/gradients/dense_2/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
q
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12* 
_output_shapes
:
*
T0
x
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read* 
_output_shapes
:
*
T0
Z
training/Adam/sub_9/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

s
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2* 
_output_shapes
:
*
T0
q
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14*
T0* 
_output_shapes
:

n
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7* 
_output_shapes
:
*
T0
[
training/Adam/Const_22Const*
_output_shapes
: *
dtype0*
valueB
 *    
[
training/Adam/Const_23Const*
_output_shapes
: *
dtype0*
valueB
 *  

%training/Adam/clip_by_value_3/MinimumMinimumtraining/Adam/add_8training/Adam/Const_23*
T0* 
_output_shapes
:


training/Adam/clip_by_value_3Maximum%training/Adam/clip_by_value_3/Minimumtraining/Adam/Const_22*
T0* 
_output_shapes
:

f
training/Adam/Sqrt_3Sqrttraining/Adam/clip_by_value_3*
T0* 
_output_shapes
:

Z
training/Adam/add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
r
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y* 
_output_shapes
:
*
T0
x
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9* 
_output_shapes
:
*
T0
t
training/Adam/sub_10Subdense_1/kernel/readtraining/Adam/truediv_3* 
_output_shapes
:
*
T0
Đ
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*+
_class!
loc:@training/Adam/Variable_2*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0
Ň
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
T0*,
_class"
 loc:@training/Adam/Variable_10* 
_output_shapes
:
*
validate_shape(*
use_locking(
˝
training/Adam/Assign_8Assigndense_1/kerneltraining/Adam/sub_10*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel
r
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes	
:
[
training/Adam/sub_11/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_17Multraining/Adam/sub_118training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_10Addtraining/Adam/mul_16training/Adam/mul_17*
_output_shapes	
:*
T0
s
training/Adam/mul_18MulAdam/beta_2/readtraining/Adam/Variable_11/read*
_output_shapes	
:*
T0
[
training/Adam/sub_12/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes	
:
m
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
_output_shapes	
:*
T0
j
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
T0*
_output_shapes	
:
[
training/Adam/Const_24Const*
_output_shapes
: *
dtype0*
valueB
 *    
[
training/Adam/Const_25Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_25*
T0*
_output_shapes	
:

training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_24*
T0*
_output_shapes	
:
a
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
T0*
_output_shapes	
:
[
training/Adam/add_12/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
o
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
_output_shapes	
:*
T0
t
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
_output_shapes	
:*
T0
m
training/Adam/sub_13Subdense_1/bias/readtraining/Adam/truediv_4*
_output_shapes	
:*
T0
Ě
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
use_locking(*
_output_shapes	
:*
validate_shape(*+
_class!
loc:@training/Adam/Variable_3*
T0
Ď
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:*,
_class"
 loc:@training/Adam/Variable_11
ľ
training/Adam/Assign_11Assigndense_1/biastraining/Adam/sub_13*
_class
loc:@dense_1/bias*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
w
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0* 
_output_shapes
:

[
training/Adam/sub_14/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

r
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22*
T0* 
_output_shapes
:

x
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0* 
_output_shapes
:

[
training/Adam/sub_15/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
training/Adam/sub_15Subtraining/Adam/sub_15/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_4Square4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

t
training/Adam/mul_24Multraining/Adam/sub_15training/Adam/Square_4*
T0* 
_output_shapes
:

r
training/Adam/add_14Addtraining/Adam/mul_23training/Adam/mul_24*
T0* 
_output_shapes
:

o
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13* 
_output_shapes
:
*
T0
[
training/Adam/Const_26Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_27Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_27* 
_output_shapes
:
*
T0

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_26* 
_output_shapes
:
*
T0
f
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5* 
_output_shapes
:
*
T0
[
training/Adam/add_15/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
t
training/Adam/add_15Addtraining/Adam/Sqrt_5training/Adam/add_15/y* 
_output_shapes
:
*
T0
y
training/Adam/truediv_5RealDivtraining/Adam/mul_25training/Adam/add_15* 
_output_shapes
:
*
T0
t
training/Adam/sub_16Subdense_2/kernel/readtraining/Adam/truediv_5*
T0* 
_output_shapes
:

Ň
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(* 
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_4*
validate_shape(*
T0
Ô
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_12*
T0*
use_locking(* 
_output_shapes
:

ž
training/Adam/Assign_14Assigndense_2/kerneltraining/Adam/sub_16*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*!
_class
loc:@dense_2/kernel
r
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes	
:
[
training/Adam/sub_17/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_27Multraining/Adam/sub_178training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
m
training/Adam/add_16Addtraining/Adam/mul_26training/Adam/mul_27*
_output_shapes	
:*
T0
s
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
_output_shapes	
:*
T0
[
training/Adam/sub_18/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
o
training/Adam/mul_29Multraining/Adam/sub_18training/Adam/Square_5*
T0*
_output_shapes	
:
m
training/Adam/add_17Addtraining/Adam/mul_28training/Adam/mul_29*
T0*
_output_shapes	
:
j
training/Adam/mul_30Multraining/Adam/multraining/Adam/add_16*
_output_shapes	
:*
T0
[
training/Adam/Const_28Const*
valueB
 *    *
_output_shapes
: *
dtype0
[
training/Adam/Const_29Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_29*
_output_shapes	
:*
T0

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_28*
T0*
_output_shapes	
:
a
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes	
:
[
training/Adam/add_18/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
o
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes	
:
t
training/Adam/truediv_6RealDivtraining/Adam/mul_30training/Adam/add_18*
T0*
_output_shapes	
:
m
training/Adam/sub_19Subdense_2/bias/readtraining/Adam/truediv_6*
_output_shapes	
:*
T0
Í
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
T0*
use_locking(*
validate_shape(*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes	
:
Ď
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
T0*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes	
:*
use_locking(*
validate_shape(
ľ
training/Adam/Assign_17Assigndense_2/biastraining/Adam/sub_19*
T0*
use_locking(*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes	
:
v
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
_output_shapes
:	*
T0
[
training/Adam/sub_20/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
q
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
_output_shapes
:	*
T0
w
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes
:	
[
training/Adam/sub_21/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	
s
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
_output_shapes
:	*
T0
q
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
_output_shapes
:	*
T0
n
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
T0*
_output_shapes
:	
[
training/Adam/Const_30Const*
dtype0*
valueB
 *    *
_output_shapes
: 
[
training/Adam/Const_31Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_31*
T0*
_output_shapes
:	

training/Adam/clip_by_value_7Maximum%training/Adam/clip_by_value_7/Minimumtraining/Adam/Const_30*
_output_shapes
:	*
T0
e
training/Adam/Sqrt_7Sqrttraining/Adam/clip_by_value_7*
_output_shapes
:	*
T0
[
training/Adam/add_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
s
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
T0*
_output_shapes
:	
x
training/Adam/truediv_7RealDivtraining/Adam/mul_35training/Adam/add_21*
_output_shapes
:	*
T0
s
training/Adam/sub_22Subdense_3/kernel/readtraining/Adam/truediv_7*
T0*
_output_shapes
:	
Ń
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	*+
_class!
loc:@training/Adam/Variable_6
Ó
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
T0*
_output_shapes
:	*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_14*
use_locking(
˝
training/Adam/Assign_20Assigndense_3/kerneltraining/Adam/sub_22*
use_locking(*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	*
validate_shape(
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
_output_shapes
:*
T0
[
training/Adam/sub_23/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_37Multraining/Adam/sub_238training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
l
training/Adam/add_22Addtraining/Adam/mul_36training/Adam/mul_37*
T0*
_output_shapes
:
r
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_15/read*
T0*
_output_shapes
:
[
training/Adam/sub_24/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
training/Adam/sub_24Subtraining/Adam/sub_24/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_7Square8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
n
training/Adam/mul_39Multraining/Adam/sub_24training/Adam/Square_7*
_output_shapes
:*
T0
l
training/Adam/add_23Addtraining/Adam/mul_38training/Adam/mul_39*
T0*
_output_shapes
:
i
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
_output_shapes
:*
T0
[
training/Adam/Const_32Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_33Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_33*
T0*
_output_shapes
:

training/Adam/clip_by_value_8Maximum%training/Adam/clip_by_value_8/Minimumtraining/Adam/Const_32*
T0*
_output_shapes
:
`
training/Adam/Sqrt_8Sqrttraining/Adam/clip_by_value_8*
_output_shapes
:*
T0
[
training/Adam/add_24/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
n
training/Adam/add_24Addtraining/Adam/Sqrt_8training/Adam/add_24/y*
_output_shapes
:*
T0
s
training/Adam/truediv_8RealDivtraining/Adam/mul_40training/Adam/add_24*
T0*
_output_shapes
:
l
training/Adam/sub_25Subdense_3/bias/readtraining/Adam/truediv_8*
T0*
_output_shapes
:
Ě
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
_output_shapes
:
Î
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:*
T0
´
training/Adam/Assign_23Assigndense_3/biastraining/Adam/sub_25*
validate_shape(*
use_locking(*
_class
loc:@dense_3/bias*
_output_shapes
:*
T0
ˇ
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/Adam/AssignAdd^training/Adam/Assign^training/Adam/Assign_1^training/Adam/Assign_2^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23
0

group_depsNoOp	^loss/mul^metrics/acc/Mean

IsVariableInitializedIsVariableInitializeddense/kernel*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel

IsVariableInitialized_1IsVariableInitialized
dense/bias*
dtype0*
_class
loc:@dense/bias*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_3IsVariableInitializeddense_1/bias*
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
: 

IsVariableInitialized_4IsVariableInitializeddense_2/kernel*
_output_shapes
: *
dtype0*!
_class
loc:@dense_2/kernel

IsVariableInitialized_5IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
: *
dtype0

IsVariableInitialized_6IsVariableInitializeddense_3/kernel*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0

IsVariableInitialized_7IsVariableInitializeddense_3/bias*
_output_shapes
: *
dtype0*
_class
loc:@dense_3/bias

IsVariableInitialized_8IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
_output_shapes
: *
dtype0	
z
IsVariableInitialized_9IsVariableInitializedAdam/lr*
dtype0*
_class
loc:@Adam/lr*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
_output_shapes
: *
dtype0*
_class
loc:@Adam/beta_1

IsVariableInitialized_11IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
dtype0*
_output_shapes
: 

IsVariableInitialized_12IsVariableInitialized
Adam/decay*
_output_shapes
: *
dtype0*
_class
loc:@Adam/decay

IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
_output_shapes
: *
dtype0

IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_1*
dtype0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes
: 

IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_2*
dtype0*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes
: 

IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_3*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes
: *
dtype0

IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_5

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_6*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes
: *
dtype0

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
: *
dtype0

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*
dtype0*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
: 

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_9*
_output_shapes
: *
dtype0*+
_class!
loc:@training/Adam/Variable_9

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_10*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*
dtype0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_12*
_output_shapes
: *
dtype0*,
_class"
 loc:@training/Adam/Variable_12

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_13*
dtype0

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
dtype0*
_output_shapes
: 

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_15*
dtype0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
: 
Ě
initNoOp^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign ^training/Adam/Variable_2/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign""
trainable_variablesőň
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0
m
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:0
\
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:0
m
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:0
\
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:0
d
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:0
D
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:0
T
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:0
T
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:0
P
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:0
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/Const_2:0
w
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/Const_3:0
w
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/Const_4:0
w
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/Const_5:0
w
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/Const_6:0
w
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/Const_7:0
w
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/Const_8:0
w
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/Const_9:0
x
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/Const_10:0
x
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/Const_11:0
{
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/Const_12:0
{
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/Const_13:0
{
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/Const_14:0
{
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/Const_15:0
{
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/Const_16:0
{
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/Const_17:0"
	variablesőň
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0
m
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:0
\
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:0
m
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:0
\
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:0
d
Adam/iterations:0Adam/iterations/AssignAdam/iterations/read:02Adam/iterations/initial_value:0
D
	Adam/lr:0Adam/lr/AssignAdam/lr/read:02Adam/lr/initial_value:0
T
Adam/beta_1:0Adam/beta_1/AssignAdam/beta_1/read:02Adam/beta_1/initial_value:0
T
Adam/beta_2:0Adam/beta_2/AssignAdam/beta_2/read:02Adam/beta_2/initial_value:0
P
Adam/decay:0Adam/decay/AssignAdam/decay/read:02Adam/decay/initial_value:0
q
training/Adam/Variable:0training/Adam/Variable/Assigntraining/Adam/Variable/read:02training/Adam/Const_2:0
w
training/Adam/Variable_1:0training/Adam/Variable_1/Assigntraining/Adam/Variable_1/read:02training/Adam/Const_3:0
w
training/Adam/Variable_2:0training/Adam/Variable_2/Assigntraining/Adam/Variable_2/read:02training/Adam/Const_4:0
w
training/Adam/Variable_3:0training/Adam/Variable_3/Assigntraining/Adam/Variable_3/read:02training/Adam/Const_5:0
w
training/Adam/Variable_4:0training/Adam/Variable_4/Assigntraining/Adam/Variable_4/read:02training/Adam/Const_6:0
w
training/Adam/Variable_5:0training/Adam/Variable_5/Assigntraining/Adam/Variable_5/read:02training/Adam/Const_7:0
w
training/Adam/Variable_6:0training/Adam/Variable_6/Assigntraining/Adam/Variable_6/read:02training/Adam/Const_8:0
w
training/Adam/Variable_7:0training/Adam/Variable_7/Assigntraining/Adam/Variable_7/read:02training/Adam/Const_9:0
x
training/Adam/Variable_8:0training/Adam/Variable_8/Assigntraining/Adam/Variable_8/read:02training/Adam/Const_10:0
x
training/Adam/Variable_9:0training/Adam/Variable_9/Assigntraining/Adam/Variable_9/read:02training/Adam/Const_11:0
{
training/Adam/Variable_10:0 training/Adam/Variable_10/Assign training/Adam/Variable_10/read:02training/Adam/Const_12:0
{
training/Adam/Variable_11:0 training/Adam/Variable_11/Assign training/Adam/Variable_11/read:02training/Adam/Const_13:0
{
training/Adam/Variable_12:0 training/Adam/Variable_12/Assign training/Adam/Variable_12/read:02training/Adam/Const_14:0
{
training/Adam/Variable_13:0 training/Adam/Variable_13/Assign training/Adam/Variable_13/read:02training/Adam/Const_15:0
{
training/Adam/Variable_14:0 training/Adam/Variable_14/Assign training/Adam/Variable_14/read:02training/Adam/Const_16:0
{
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/Const_17:0ůŁé#       çÎř	{iţ0Ń×A*


accá	?˝ń       ŁK"	Ďjţ0Ń×A*

lossdH´?ˇŮ4!       	Škţ0Ń×A*

val_accę?ýÖV       ČÁ	lţ0Ń×A*

val_lossIw?ŃÖ-       ń(	Ż\Ř
1Ń×A*


acc7+?űëë       Ř-	˙]Ř
1Ń×A*

lossÂŘ~?ź:Ž˛       `/ß#	sgŘ
1Ń×A*

val_accž^/?Ű~.       ŮÜ2	V}Ř
1Ń×A*

val_loss÷c?÷ď÷       ń(	g(1Ń×A*


acc/ź9?e       Ř-	­21Ń×A*

lossQ?Đn˝U       `/ß#	41Ń×A*

val_accC?űăw       ŮÜ2	61Ń×A*

val_lossç7>?Ti"       ń(	nQ*1Ń×A*


accĐ*E?	H^       Ř-	ŇR*1Ń×A*

lossGM/?7raw       `/ß#	ľS*1Ń×A*

val_acc!;H?fÜ       ŮÜ2	a*1Ń×A*

val_lossÝ"?+?t       ń(	ś÷í61Ń×A*


acc|O?ąşĎ}       Ř-	ůí61Ń×A*

lossfL?~ŘQ       `/ß#	ôůí61Ń×A*

val_accD÷P?kNÉ       ŮÜ2	Ęúí61Ń×A*

val_lossË_
?neŐ       ń(	ČĄC1Ń×A*


acceW?ň       Ř-	ÉĄC1Ń×A*

loss˝5ő>íÍç       `/ß#	ĘĄC1Ń×A*

val_acck`?Ůľě       ŮÜ2	ËĄC1Ń×A*

val_loss­>ß>ű2üQ       ń(	ž˘ŇP1Ń×A*


acc¤4_?Ű&ČŻ       Ř-	łŇP1Ń×A*

lossë$Í>NcĘ       `/ß#	)´ŇP1Ń×A*

val_accc?Sâą       ŮÜ2	4ťŇP1Ń×A*

val_lossłĽş>ŚCőů       ń(	Ob]1Ń×A*


accŃ/e?řĐ÷       Ř-	Qb]1Ń×A*

lossÂÚ­>ˇ´6Î       `/ß#	aSb]1Ń×A*

val_accľ/m?8&ăV       ŮÜ2	;Ub]1Ń×A*

val_lossyž>^?t       ń(	~súi1Ń×A*


acc	j?iŐŇ       Ř-	)uúi1Ń×A*

loss(?>5é9       `/ß#	!vúi1Ń×A*

val_acc˝3o?Ź8ř       ŮÜ2	wúi1Ń×A*

val_lossá-|>2ň       ń(	w1Ń×A	*


acc=ám?ßí{       Ř-	ww1Ń×A	*

lossŃ.|>őgÖ>       `/ß#	Zw1Ń×A	*

val_accşn?Š4Ťd       ŮÜ2	8w1Ń×A	*

val_loss(ťk>m5ię       ń(	"9Ô1Ń×A
*


accÄ˝p?ś¨       Ř-	~:Ô1Ń×A
*

loss­čX>Ő|&       `/ß#	m;Ô1Ń×A
*

val_accľ/m?pąÖ       ŮÜ2	S<Ô1Ń×A
*

val_lossŻđ[>&ľŔ       ń(	őŰ1Ń×A*


accËns?Xë<       Ř-	÷Ű1Ń×A*

lossľS:>KŻ+       `/ß#	ůŰ1Ń×A*

val_accÍ;s?jz|       ŮÜ2	ŠúŰ1Ń×A*

val_lossJ>5 ­ž       ń(	Aç1Ń×A*


accüUu?ç9óA       Ř-	lBç1Ń×A*

lossĹ!>W	6       `/ß#	_Cç1Ń×A*

val_accÍ;s?bÎŐ       ŮÜ2	=Dç1Ń×A*

val_loss>%`B       ń(	P[Ť1Ń×A*


acc3w?ęÖ7ô       Ř-	°[Ť1Ń×A*

lossO>&cö       `/ß#	m[Ť1Ń×A*

val_accęz?ŰŐ´       ŮÜ2	![Ť1Ń×A*

val_loss5(Ů=_ęßŇ       ń(	˝Ľ¸1Ń×A*


accĘy?vü       Ř-	sžĽ¸1Ń×A*

lossű°ó={ŢW       `/ß#	UżĽ¸1Ń×A*

val_accŕďw?7Ô       ŮÜ2	3ŔĽ¸1Ń×A*

val_loss/Á>á       ń(	1ëÇ1Ń×A*


acc5îy?ßÎ       Ř-	1íÇ1Ń×A*

lossvĐÖ=9F.       `/ß#	ŻîÇ1Ń×A*

val_accóŁ|?mČŘ-       ŮÜ2	$đÇ1Ń×A*

val_lossŤüť=        ń(	:YŐ1Ń×A*


accz?ŤĺP×       Ř-	[Ő1Ń×A*

lossaÂ˝=ĽÜżâ       `/ß#	S\Ő1Ń×A*

val_accęz?Ľ@aŠ       ŮÜ2	y]Ő1Ń×A*

val_loss8Ú=D2v       ń(	(ňłâ1Ń×A*


accg'{?Ýşý       Ř-	Ąółâ1Ń×A*

lossěŤ=1ď")       `/ß#	ôłâ1Ń×A*

val_accĺGy?ńbş       ŮÜ2	őłâ1Ń×A*

val_lossäI˘=@úęK       ń(	¤đ1Ń×A*


accč{?KyT       Ř-	eđ1Ń×A*

loss7K=ÂŐ#!       `/ß#	°đ1Ń×A*

val_accóŁ|?pĂőť       ŮÜ2	 đ1Ń×A*

val_loss,f=|Ň	       ń(	Ýjţ1Ń×A*


accLi|?X#ÍT       Ř-	5jţ1Ń×A*

loss[=ď;˛       `/ß#	jţ1Ń×A*

val_accđ÷{?Î?G       ŮÜ2	ňjţ1Ń×A*

val_lossŚo=QRś