       ŁK"	  Ŕ2Ń×Abrain.Event:2ÝŔ9¤Ţć     ŞÄú	Kwď2Ń×A"ŃÍ
p
dense_1_inputPlaceholder*
shape:˙˙˙˙˙˙˙˙˙1*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1

-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:*
valueB"1      

+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: *
valueB
 *<ž

+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *<>*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel
ć
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_class
loc:@dense/kernel*
seed2 *
T0*
_output_shapes
:	1*
dtype0*

seed 
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
_class
loc:@dense/kernel*
T0
á
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	1*
T0*
_class
loc:@dense/kernel
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
:	1
Ł
dense/kernel
VariableV2*
dtype0*
shared_name *
_output_shapes
:	1*
	container *
_class
loc:@dense/kernel*
shape:	1
Č
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
T0*
use_locking(*
_output_shapes
:	1*
validate_shape(
v
dense/kernel/readIdentitydense/kernel*
_output_shapes
:	1*
T0*
_class
loc:@dense/kernel

dense/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@dense/bias*
_output_shapes	
:


dense/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes	
:*
shape:*
_class
loc:@dense/bias
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
_output_shapes	
:*
_class
loc:@dense/bias*
use_locking(*
T0*
validate_shape(
l
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias*
_output_shapes	
:

dense/MatMulMatMuldense_1_inputdense/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
transpose_a( *
T0

dense/BiasAddBiasAdddense/MatMuldense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
T

dense/ReluReludense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_1/kernel*
_output_shapes
:*
valueB"      *
dtype0

-dense_1/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *   ž

-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
valueB
 *   >
í
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*

seed *
seed2 *
dtype0*
T0*!
_class
loc:@dense_1/kernel
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
ę
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
T0
Ü
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
T0
Š
dense_1/kernel
VariableV2*
dtype0*
shared_name *
	container * 
_output_shapes
:
*!
_class
loc:@dense_1/kernel*
shape:

Ń
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(*!
_class
loc:@dense_1/kernel
}
dense_1/kernel/readIdentitydense_1/kernel* 
_output_shapes
:
*
T0*!
_class
loc:@dense_1/kernel

dense_1/bias/Initializer/zerosConst*
_output_shapes	
:*
_class
loc:@dense_1/bias*
dtype0*
valueB*    

dense_1/bias
VariableV2*
	container *
shape:*
dtype0*
shared_name *
_class
loc:@dense_1/bias*
_output_shapes	
:
ť
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
T0*
use_locking(*
_class
loc:@dense_1/bias
r
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes	
:

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0*
transpose_a( 

dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
X
dense_2/ReluReludense_2/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*!
_class
loc:@dense_2/kernel*
valueB"      *
dtype0

-dense_2/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
valueB
 *óľ˝*
dtype0

-dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *óľ=*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
í
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
*
T0*
seed2 *

seed *!
_class
loc:@dense_2/kernel*
dtype0
Ö
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
ę
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_2/kernel* 
_output_shapes
:

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
	container * 
_output_shapes
:
*
shape:
*
dtype0*
shared_name *!
_class
loc:@dense_2/kernel
Ń
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:

}
dense_2/kernel/readIdentitydense_2/kernel*
T0* 
_output_shapes
:
*!
_class
loc:@dense_2/kernel

dense_2/bias/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
_class
loc:@dense_2/bias*
dtype0

dense_2/bias
VariableV2*
shape:*
dtype0*
_class
loc:@dense_2/bias*
_output_shapes	
:*
	container *
shared_name 
ť
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
use_locking(*
validate_shape(*
_class
loc:@dense_2/bias*
T0*
_output_shapes	
:
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
transpose_a( *
transpose_b( 

dense_3/BiasAddBiasAdddense_3/MatMuldense_2/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
dense_3/ReluReludense_3/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*!
_class
loc:@dense_3/kernel*
valueB"      

-dense_3/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_3/kernel*
dtype0*
valueB
 *żđÚ˝*
_output_shapes
: 

-dense_3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
dtype0*
valueB
 *żđÚ=
ě
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
seed2 *
_output_shapes
:	*!
_class
loc:@dense_3/kernel*
dtype0*
T0*

seed 
Ö
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_3/kernel
é
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
_output_shapes
:	*
T0*!
_class
loc:@dense_3/kernel
Ű
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_3/kernel*
_output_shapes
:	*
T0
§
dense_3/kernel
VariableV2*
dtype0*!
_class
loc:@dense_3/kernel*
	container *
shape:	*
_output_shapes
:	*
shared_name 
Đ
dense_3/kernel/AssignAssigndense_3/kernel)dense_3/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes
:	*
T0*
use_locking(*!
_class
loc:@dense_3/kernel
|
dense_3/kernel/readIdentitydense_3/kernel*!
_class
loc:@dense_3/kernel*
_output_shapes
:	*
T0

dense_3/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_3/bias*
dtype0*
_output_shapes
:

dense_3/bias
VariableV2*
shape:*
shared_name *
_class
loc:@dense_3/bias*
dtype0*
	container *
_output_shapes
:
ş
dense_3/bias/AssignAssigndense_3/biasdense_3/bias/Initializer/zeros*
_class
loc:@dense_3/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:*
T0*
_class
loc:@dense_3/bias

dense_4/MatMulMatMuldense_3/Reludense_3/kernel/read*
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 

dense_4/BiasAddBiasAdddense_4/MatMuldense_3/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
]
dense_4/SoftmaxSoftmaxdense_4/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
Adam/iterations/initial_valueConst*
value	B	 R *
_output_shapes
: *
dtype0	
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
loc:@Adam/iterations*
T0	*
validate_shape(*
_output_shapes
: *
use_locking(
v
Adam/iterations/readIdentityAdam/iterations*"
_class
loc:@Adam/iterations*
T0	*
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
: *
shape: *
shared_name *
	container 

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_class
loc:@Adam/lr*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
^
Adam/lr/readIdentityAdam/lr*
_class
loc:@Adam/lr*
T0*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
o
Adam/beta_1
VariableV2*
shared_name *
_output_shapes
: *
shape: *
	container *
dtype0
Ž
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
use_locking(*
_output_shapes
: *
validate_shape(*
_class
loc:@Adam/beta_1*
T0
j
Adam/beta_1/readIdentityAdam/beta_1*
T0*
_output_shapes
: *
_class
loc:@Adam/beta_1
^
Adam/beta_2/initial_valueConst*
dtype0*
valueB
 *wž?*
_output_shapes
: 
o
Adam/beta_2
VariableV2*
shape: *
_output_shapes
: *
dtype0*
	container *
shared_name 
Ž
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_class
loc:@Adam/beta_2*
use_locking(*
_output_shapes
: *
T0*
validate_shape(
j
Adam/beta_2/readIdentityAdam/beta_2*
T0*
_output_shapes
: *
_class
loc:@Adam/beta_2
]
Adam/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
n

Adam/decay
VariableV2*
_output_shapes
: *
shape: *
shared_name *
dtype0*
	container 
Ş
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
validate_shape(*
_class
loc:@Adam/decay*
_output_shapes
: *
T0*
use_locking(
g
Adam/decay/readIdentity
Adam/decay*
_class
loc:@Adam/decay*
T0*
_output_shapes
: 

dense_4_targetPlaceholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0
q
dense_4_sample_weightsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
\
loss/dense_4_loss/ConstConst*
_output_shapes
: *
valueB
 *żÖ3*
dtype0
\
loss/dense_4_loss/sub/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
o
loss/dense_4_loss/subSubloss/dense_4_loss/sub/xloss/dense_4_loss/Const*
_output_shapes
: *
T0
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
loss/dense_4_loss/LogLogloss/dense_4_loss/clip_by_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
loss/dense_4_loss/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

loss/dense_4_loss/ReshapeReshapedense_4_targetloss/dense_4_loss/Reshape/shape*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
v
loss/dense_4_loss/CastCastloss/dense_4_loss/Reshape*

DstT0	*

SrcT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
r
!loss/dense_4_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
 
loss/dense_4_loss/Reshape_1Reshapeloss/dense_4_loss/Log!loss/dense_4_loss/Reshape_1/shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

;loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/dense_4_loss/Cast*
out_type0*
_output_shapes
:*
T0	

Yloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss/dense_4_loss/Reshape_1loss/dense_4_loss/Cast*
T0*
Tlabels0	*6
_output_shapes$
":˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
k
(loss/dense_4_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
î
loss/dense_4_loss/MeanMeanYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(loss/dense_4_loss/Mean/reduction_indices*
T0*
	keep_dims( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

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
loss/dense_4_loss/NotEqualNotEqualdense_4_sample_weightsloss/dense_4_loss/NotEqual/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
y
loss/dense_4_loss/Cast_1Castloss/dense_4_loss/NotEqual*

SrcT0
*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
loss/dense_4_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 

loss/dense_4_loss/Mean_1Meanloss/dense_4_loss/Cast_1loss/dense_4_loss/Const_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0

loss/dense_4_loss/truedivRealDivloss/dense_4_loss/mulloss/dense_4_loss/Mean_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
loss/dense_4_loss/Const_2Const*
dtype0*
_output_shapes
:*
valueB: 

loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_2*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
O

loss/mul/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
metrics/acc/MaxMaxdense_4_target!metrics/acc/Max/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0*
	keep_dims( 
g
metrics/acc/ArgMax/dimensionConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 

metrics/acc/ArgMaxArgMaxdense_4/Softmaxmetrics/acc/ArgMax/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
output_type0	*
T0*

Tidx0
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
metrics/acc/ConstConst*
_output_shapes
:*
valueB: *
dtype0
}
metrics/acc/MeanMeanmetrics/acc/Cast_1metrics/acc/Const*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
}
training/Adam/gradients/ShapeConst*
dtype0*
_output_shapes
: *
_class
loc:@loss/mul*
valueB 

!training/Adam/gradients/grad_ys_0Const*
_class
loc:@loss/mul*
_output_shapes
: *
valueB
 *  ?*
dtype0
¤
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_output_shapes
: *
_class
loc:@loss/mul
Ś
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_2*
_class
loc:@loss/mul*
T0*
_output_shapes
: 

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss/mul
ş
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0*+
_class!
loc:@loss/dense_4_loss/Mean_2

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape/shape*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
:*
Tshape0*
T0
Á
;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ShapeShapeloss/dense_4_loss/truediv*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
:*
T0*
out_type0
Ť
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*+
_class!
loc:@loss/dense_4_loss/Mean_2
Ă
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_1Shapeloss/dense_4_loss/truediv*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0*
out_type0
­
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_2Const*
valueB *+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
: *
dtype0
˛
;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_2
Š
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
´
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0*+
_class!
loc:@loss/dense_4_loss/Mean_2
­
<training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const_1*
T0*
_output_shapes
: *

Tidx0*+
_class!
loc:@loss/dense_4_loss/Mean_2*
	keep_dims( 
Ž
?training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0*+
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
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
: *
T0
ß
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/floordiv*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
: *

DstT0*

SrcT0

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
out_type0*
_output_shapes
:
Ż
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*,
_class"
 loc:@loss/dense_4_loss/truediv*
dtype0*
valueB *
_output_shapes
: 
Î
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*,
_class"
 loc:@loss/dense_4_loss/truediv*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ţ
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truedivloss/dense_4_loss/Mean_1*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˝
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0*,
_class"
 loc:@loss/dense_4_loss/truediv
­
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0*
T0
´
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/NegNegloss/dense_4_loss/mul*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ý
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1RealDiv:training/Adam/gradients/loss/dense_4_loss/truediv_grad/Negloss/dense_4_loss/Mean_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv

@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1loss/dense_4_loss/Mean_1*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulMul=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truediv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:*,
_class"
 loc:@loss/dense_4_loss/truediv
Ś
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
_output_shapes
: *
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
Tshape0
¸
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean*(
_class
loc:@loss/dense_4_loss/mul*
T0*
_output_shapes
:*
out_type0
ş
:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1Shapedense_4_sample_weights*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:*
T0*
out_type0
ž
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul
í
6training/Adam/gradients/loss/dense_4_loss/mul_grad/mulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*(
_class
loc:@loss/dense_4_loss/mul*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Š
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/mulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0*(
_class
loc:@loss/dense_4_loss/mul

:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
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
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *(
_class
loc:@loss/dense_4_loss/mul*

Tidx0*
T0*
_output_shapes
:
Ł
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*(
_class
loc:@loss/dense_4_loss/mul
ý
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
_output_shapes
:
Ľ
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*
dtype0*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
đ
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
T0

7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
_output_shapes
: 
°
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*
valueB: *
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/startConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/deltaConst*
dtype0*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 
Ń
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*

Tidx0*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean
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
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*)
_class
loc:@loss/dense_4_loss/Mean*
N
Ş
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
Ą
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*)
_class
loc:@loss/dense_4_loss/Mean

<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordivFloorDiv9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*)
_class
loc:@loss/dense_4_loss/Mean
Ą
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*
T0*
_output_shapes
:*
Tshape0*)
_class
loc:@loss/dense_4_loss/Mean
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
:*
T0*
out_type0*)
_class
loc:@loss/dense_4_loss/Mean
ź
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*
T0*
out_type0
Ž
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
Ą
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
°
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean
Ľ
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( *)
_class
loc:@loss/dense_4_loss/Mean
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
loc:@loss/dense_4_loss/Mean*
T0*
_output_shapes
: 

>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
T0*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean
Ű
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*

DstT0

;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*)
_class
loc:@loss/dense_4_loss/Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ź
"training/Adam/gradients/zeros_like	ZerosLike[loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
Î
training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient[loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*´
message¨ĽCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
ż
training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0

training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivtraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
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
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*.
_class$
" loc:@loss/dense_4_loss/Reshape_1

=training/Adam/gradients/loss/dense_4_loss/Log_grad/Reciprocal
Reciprocalloss/dense_4_loss/clip_by_valueA^training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/Log

6training/Adam/gradients/loss/dense_4_loss/Log_grad/mulMul@training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Reshape=training/Adam/gradients/loss/dense_4_loss/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/Log*
T0
Ý
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ShapeShape'loss/dense_4_loss/clip_by_value/Minimum*
T0*
out_type0*
_output_shapes
:*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
ť
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1Const*
valueB *
_output_shapes
: *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
dtype0
î
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_2Shape6training/Adam/gradients/loss/dense_4_loss/Log_grad/mul*
_output_shapes
:*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
out_type0*
T0
Á
Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros/ConstConst*
_output_shapes
: *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
dtype0*
valueB
 *    
Ŕ
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zerosFillDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_2Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros/Const*
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Itraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_4_loss/clip_by_value/Minimumloss/dense_4_loss/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
ć
Rtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ShapeDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0
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

Tidx0*
_output_shapes
:*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0*
	keep_dims( 
É
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape*
Tshape0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ú
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Sum_1SumEtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Select_1Ttraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
ž
Ftraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
Ő
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ShapeShapedense_4/Softmax*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
_output_shapes
:*
out_type0*
T0
Ë
Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
valueB *
_output_shapes
: 

Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_2ShapeDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
out_type0*
_output_shapes
:*
T0
Ń
Ptraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros/ConstConst*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
dtype0*
_output_shapes
: *
valueB
 *    
ŕ
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros/Const*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ń
Ntraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualdense_4/Softmaxloss/dense_4_loss/sub*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ztraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
Ktraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SelectSelectNtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqualDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ReshapeJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
§
Mtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Select_1SelectNtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqualJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zerosDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
ô
Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*

Tidx0*
	keep_dims( *
T0
é
Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
ú
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
_output_shapes
:*
T0
Ţ
Ntraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0*
Tshape0*
_output_shapes
: 
ě
0training/Adam/gradients/dense_4/Softmax_grad/mulMulLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshapedense_4/Softmax*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*"
_class
loc:@dense_4/Softmax
°
Btraining/Adam/gradients/dense_4/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:*"
_class
loc:@dense_4/Softmax

0training/Adam/gradients/dense_4/Softmax_grad/SumSum0training/Adam/gradients/dense_4/Softmax_grad/mulBtraining/Adam/gradients/dense_4/Softmax_grad/Sum/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*"
_class
loc:@dense_4/Softmax*
	keep_dims( *
T0*

Tidx0
Ż
:training/Adam/gradients/dense_4/Softmax_grad/Reshape/shapeConst*"
_class
loc:@dense_4/Softmax*
dtype0*
_output_shapes
:*
valueB"˙˙˙˙   

4training/Adam/gradients/dense_4/Softmax_grad/ReshapeReshape0training/Adam/gradients/dense_4/Softmax_grad/Sum:training/Adam/gradients/dense_4/Softmax_grad/Reshape/shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*"
_class
loc:@dense_4/Softmax
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
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Softmax_grad/mul_1*
_output_shapes
:*
T0*"
_class
loc:@dense_4/BiasAdd*
data_formatNHWC

2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Softmax_grad/mul_1dense_3/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*!
_class
loc:@dense_4/MatMul*
transpose_b(*
transpose_a( 
ó
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Relu2training/Adam/gradients/dense_4/Softmax_grad/mul_1*
T0*
_output_shapes
:	*
transpose_b( *!
_class
loc:@dense_4/MatMul*
transpose_a(
Ô
2training/Adam/gradients/dense_3/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_4/MatMul_grad/MatMuldense_3/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@dense_3/Relu
Ü
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes	
:*"
_class
loc:@dense_3/BiasAdd*
T0

2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Relu_grad/ReluGraddense_2/kernel/read*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*!
_class
loc:@dense_3/MatMul*
transpose_a( *
T0
ô
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
T0*!
_class
loc:@dense_3/MatMul*
transpose_a(*
transpose_b( * 
_output_shapes
:

Ô
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@dense_2/Relu
Ü
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:*"
_class
loc:@dense_2/BiasAdd

2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0*
transpose_b(*!
_class
loc:@dense_2/MatMul
ň
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/Relu2training/Adam/gradients/dense_2/Relu_grad/ReluGrad* 
_output_shapes
:
*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul*
transpose_a(
Î
0training/Adam/gradients/dense/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_2/MatMul_grad/MatMul
dense/Relu*
_class
loc:@dense/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ö
6training/Adam/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0training/Adam/gradients/dense/Relu_grad/ReluGrad* 
_class
loc:@dense/BiasAdd*
T0*
_output_shapes	
:*
data_formatNHWC
ř
0training/Adam/gradients/dense/MatMul_grad/MatMulMatMul0training/Adam/gradients/dense/Relu_grad/ReluGraddense/kernel/read*
transpose_b(*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
T0*
_class
loc:@dense/MatMul
î
2training/Adam/gradients/dense/MatMul_grad/MatMul_1MatMuldense_1_input0training/Adam/gradients/dense/Relu_grad/ReluGrad*
transpose_b( *
T0*
_output_shapes
:	1*
_class
loc:@dense/MatMul*
transpose_a(
_
training/Adam/AssignAdd/valueConst*
_output_shapes
: *
value	B	 R*
dtype0	
Ź
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
use_locking( *"
_class
loc:@Adam/iterations*
_output_shapes
: *
T0	
`
training/Adam/CastCastAdam/iterations/read*

DstT0*
_output_shapes
: *

SrcT0	
X
training/Adam/add/yConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
: *
dtype0*
valueB
 *    
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
training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
_output_shapes
: *
T0
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
valueB	1*    *
dtype0*
_output_shapes
:	1

training/Adam/Variable
VariableV2*
_output_shapes
:	1*
	container *
shared_name *
shape:	1*
dtype0
Ô
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/Const_2*
_output_shapes
:	1*)
_class
loc:@training/Adam/Variable*
validate_shape(*
T0*
use_locking(

training/Adam/Variable/readIdentitytraining/Adam/Variable*
T0*
_output_shapes
:	1*)
_class
loc:@training/Adam/Variable
d
training/Adam/Const_3Const*
_output_shapes	
:*
dtype0*
valueB*    

training/Adam/Variable_1
VariableV2*
shared_name *
_output_shapes	
:*
dtype0*
	container *
shape:
Ö
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/Const_3*
use_locking(*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
T0

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_1
n
training/Adam/Const_4Const* 
_output_shapes
:
*
valueB
*    *
dtype0

training/Adam/Variable_2
VariableV2*
dtype0*
	container *
shape:
* 
_output_shapes
:
*
shared_name 
Ű
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/Const_4*
use_locking(*
validate_shape(* 
_output_shapes
:
*
T0*+
_class!
loc:@training/Adam/Variable_2

training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2* 
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_2*
T0
d
training/Adam/Const_5Const*
_output_shapes	
:*
dtype0*
valueB*    

training/Adam/Variable_3
VariableV2*
_output_shapes	
:*
	container *
dtype0*
shared_name *
shape:
Ö
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/Const_5*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
T0*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_3
n
training/Adam/Const_6Const*
valueB
*    *
dtype0* 
_output_shapes
:


training/Adam/Variable_4
VariableV2*
shared_name * 
_output_shapes
:
*
shape:
*
	container *
dtype0
Ű
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/Const_6*
validate_shape(*
T0* 
_output_shapes
:
*
use_locking(*+
_class!
loc:@training/Adam/Variable_4

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0*+
_class!
loc:@training/Adam/Variable_4* 
_output_shapes
:

d
training/Adam/Const_7Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_5
VariableV2*
shape:*
shared_name *
	container *
dtype0*
_output_shapes	
:
Ö
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/Const_7*
validate_shape(*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_5*
T0*
use_locking(

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_5
l
training/Adam/Const_8Const*
_output_shapes
:	*
valueB	*    *
dtype0

training/Adam/Variable_6
VariableV2*
shared_name *
shape:	*
_output_shapes
:	*
dtype0*
	container 
Ú
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/Const_8*
validate_shape(*+
_class!
loc:@training/Adam/Variable_6*
use_locking(*
_output_shapes
:	*
T0

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
_output_shapes
:	*
T0*+
_class!
loc:@training/Adam/Variable_6
b
training/Adam/Const_9Const*
dtype0*
valueB*    *
_output_shapes
:

training/Adam/Variable_7
VariableV2*
shared_name *
_output_shapes
:*
dtype0*
shape:*
	container 
Ő
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/Const_9*
use_locking(*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:*
validate_shape(*
T0

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:*
T0
m
training/Adam/Const_10Const*
dtype0*
_output_shapes
:	1*
valueB	1*    

training/Adam/Variable_8
VariableV2*
shape:	1*
_output_shapes
:	1*
	container *
dtype0*
shared_name 
Ű
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/Const_10*
T0*+
_class!
loc:@training/Adam/Variable_8*
use_locking(*
validate_shape(*
_output_shapes
:	1

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
T0*
_output_shapes
:	1
e
training/Adam/Const_11Const*
dtype0*
_output_shapes	
:*
valueB*    

training/Adam/Variable_9
VariableV2*
_output_shapes	
:*
shared_name *
shape:*
dtype0*
	container 
×
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/Const_11*
use_locking(*+
_class!
loc:@training/Adam/Variable_9*
validate_shape(*
T0*
_output_shapes	
:

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
T0*
_output_shapes	
:
o
training/Adam/Const_12Const* 
_output_shapes
:
*
dtype0*
valueB
*    

training/Adam/Variable_10
VariableV2* 
_output_shapes
:
*
shape:
*
shared_name *
dtype0*
	container 
ß
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/Const_12* 
_output_shapes
:
*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_10*
T0*
use_locking(

training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10*
T0* 
_output_shapes
:
*,
_class"
 loc:@training/Adam/Variable_10
e
training/Adam/Const_13Const*
dtype0*
valueB*    *
_output_shapes	
:

training/Adam/Variable_11
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_output_shapes	
:
Ú
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/Const_13*,
_class"
 loc:@training/Adam/Variable_11*
use_locking(*
validate_shape(*
_output_shapes	
:*
T0

training/Adam/Variable_11/readIdentitytraining/Adam/Variable_11*
T0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes	
:
o
training/Adam/Const_14Const*
dtype0* 
_output_shapes
:
*
valueB
*    

training/Adam/Variable_12
VariableV2*
	container *
shared_name *
shape:
*
dtype0* 
_output_shapes
:

ß
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/Const_14* 
_output_shapes
:
*,
_class"
 loc:@training/Adam/Variable_12*
validate_shape(*
use_locking(*
T0

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12* 
_output_shapes
:
*
T0
e
training/Adam/Const_15Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_13
VariableV2*
	container *
dtype0*
shape:*
shared_name *
_output_shapes	
:
Ú
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/Const_15*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*,
_class"
 loc:@training/Adam/Variable_13
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
VariableV2*
dtype0*
shape:	*
_output_shapes
:	*
shared_name *
	container 
Ţ
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/Const_16*
_output_shapes
:	*
T0*
use_locking(*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_14

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*,
_class"
 loc:@training/Adam/Variable_14*
T0*
_output_shapes
:	
c
training/Adam/Const_17Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_15
VariableV2*
shared_name *
dtype0*
shape:*
	container *
_output_shapes
:
Ů
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/Const_17*
T0*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*,
_class"
 loc:@training/Adam/Variable_15*
_output_shapes
:
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
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
_output_shapes
: *
T0
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
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
T0*
_output_shapes
:	1
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
training/Adam/Const_18Const*
dtype0*
valueB
 *    *
_output_shapes
: 
[
training/Adam/Const_19Const*
_output_shapes
: *
dtype0*
valueB
 *  
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
training/Adam/add_3/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
q
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes
:	1
v
training/Adam/truediv_1RealDivtraining/Adam/mul_5training/Adam/add_3*
_output_shapes
:	1*
T0
p
training/Adam/sub_4Subdense/kernel/readtraining/Adam/truediv_1*
_output_shapes
:	1*
T0
É
training/Adam/AssignAssigntraining/Adam/Variabletraining/Adam/add_1*
validate_shape(*
T0*
_output_shapes
:	1*)
_class
loc:@training/Adam/Variable*
use_locking(
Ď
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
T0*
_output_shapes
:	1*
validate_shape(*
use_locking(*+
_class!
loc:@training/Adam/Variable_8
ˇ
training/Adam/Assign_2Assigndense/kerneltraining/Adam/sub_4*
T0*
_class
loc:@dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	1
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
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_7Multraining/Adam/sub_56training/Adam/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
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
training/Adam/sub_6/xConst*
_output_shapes
: *
valueB
 *  ?*
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
training/Adam/add_5Addtraining/Adam/mul_8training/Adam/mul_9*
T0*
_output_shapes	
:
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
: *
valueB
 *  *
dtype0

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_21*
T0*
_output_shapes	
:

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_20*
T0*
_output_shapes	
:
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
training/Adam/sub_7Subdense/bias/readtraining/Adam/truediv_2*
_output_shapes	
:*
T0
Ë
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
T0*
_output_shapes	
:*
use_locking(
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
dense/biastraining/Adam/sub_7*
validate_shape(*
use_locking(*
_class
loc:@dense/bias*
T0*
_output_shapes	
:
w
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0* 
_output_shapes
:

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
training/Adam/add_7Addtraining/Adam/mul_11training/Adam/mul_12* 
_output_shapes
:
*
T0
x
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0* 
_output_shapes
:

Z
training/Adam/sub_9/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

s
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0* 
_output_shapes
:

q
training/Adam/add_8Addtraining/Adam/mul_13training/Adam/mul_14* 
_output_shapes
:
*
T0
n
training/Adam/mul_15Multraining/Adam/multraining/Adam/add_7*
T0* 
_output_shapes
:

[
training/Adam/Const_22Const*
valueB
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_23Const*
dtype0*
valueB
 *  *
_output_shapes
: 
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
training/Adam/sub_10Subdense_1/kernel/readtraining/Adam/truediv_3* 
_output_shapes
:
*
T0
Đ
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7*
use_locking(*
T0*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(* 
_output_shapes
:

Ň
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_10
˝
training/Adam/Assign_8Assigndense_1/kerneltraining/Adam/sub_10* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel*
validate_shape(*
T0*
use_locking(
r
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes	
:
[
training/Adam/sub_11/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_11Subtraining/Adam/sub_11/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_17Multraining/Adam/sub_118training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
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
training/Adam/sub_12/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
training/Adam/sub_12Subtraining/Adam/sub_12/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_3Square8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
o
training/Adam/mul_19Multraining/Adam/sub_12training/Adam/Square_3*
T0*
_output_shapes	
:
m
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes	
:
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
training/Adam/Const_25Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_4/MinimumMinimumtraining/Adam/add_11training/Adam/Const_25*
T0*
_output_shapes	
:

training/Adam/clip_by_value_4Maximum%training/Adam/clip_by_value_4/Minimumtraining/Adam/Const_24*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes	
:*
T0
[
training/Adam/add_12/yConst*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
o
training/Adam/add_12Addtraining/Adam/Sqrt_4training/Adam/add_12/y*
T0*
_output_shapes	
:
t
training/Adam/truediv_4RealDivtraining/Adam/mul_20training/Adam/add_12*
T0*
_output_shapes	
:
m
training/Adam/sub_13Subdense_1/bias/readtraining/Adam/truediv_4*
_output_shapes	
:*
T0
Ě
training/Adam/Assign_9Assigntraining/Adam/Variable_3training/Adam/add_10*
validate_shape(*
T0*
use_locking(*+
_class!
loc:@training/Adam/Variable_3*
_output_shapes	
:
Ď
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_11*
T0*
_output_shapes	
:
ľ
training/Adam/Assign_11Assigndense_1/biastraining/Adam/sub_13*
use_locking(*
_output_shapes	
:*
_class
loc:@dense_1/bias*
validate_shape(*
T0
w
training/Adam/mul_21MulAdam/beta_1/readtraining/Adam/Variable_4/read*
T0* 
_output_shapes
:

[
training/Adam/sub_14/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
r
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22* 
_output_shapes
:
*
T0
x
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read*
T0* 
_output_shapes
:

[
training/Adam/sub_15/xConst*
valueB
 *  ?*
dtype0*
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
training/Adam/Const_26Const*
_output_shapes
: *
dtype0*
valueB
 *    
[
training/Adam/Const_27Const*
_output_shapes
: *
dtype0*
valueB
 *  

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_27* 
_output_shapes
:
*
T0

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_26*
T0* 
_output_shapes
:

f
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5* 
_output_shapes
:
*
T0
[
training/Adam/add_15/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
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
training/Adam/sub_16Subdense_2/kernel/readtraining/Adam/truediv_5* 
_output_shapes
:
*
T0
Ň
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*
use_locking(* 
_output_shapes
:
*
validate_shape(*+
_class!
loc:@training/Adam/Variable_4*
T0
Ô
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14* 
_output_shapes
:
*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_12*
use_locking(*
T0
ž
training/Adam/Assign_14Assigndense_2/kerneltraining/Adam/sub_16*!
_class
loc:@dense_2/kernel*
validate_shape(*
use_locking(* 
_output_shapes
:
*
T0
r
training/Adam/mul_26MulAdam/beta_1/readtraining/Adam/Variable_5/read*
T0*
_output_shapes	
:
[
training/Adam/sub_17/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_17Subtraining/Adam/sub_17/xAdam/beta_1/read*
_output_shapes
: *
T0
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
training/Adam/mul_28MulAdam/beta_2/readtraining/Adam/Variable_13/read*
T0*
_output_shapes	
:
[
training/Adam/sub_18/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
f
training/Adam/sub_18Subtraining/Adam/sub_18/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_5Square8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
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
 *    *
dtype0*
_output_shapes
: 
[
training/Adam/Const_29Const*
dtype0*
valueB
 *  *
_output_shapes
: 

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_29*
_output_shapes	
:*
T0

training/Adam/clip_by_value_6Maximum%training/Adam/clip_by_value_6/Minimumtraining/Adam/Const_28*
_output_shapes	
:*
T0
a
training/Adam/Sqrt_6Sqrttraining/Adam/clip_by_value_6*
T0*
_output_shapes	
:
[
training/Adam/add_18/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
o
training/Adam/add_18Addtraining/Adam/Sqrt_6training/Adam/add_18/y*
T0*
_output_shapes	
:
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
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(*+
_class!
loc:@training/Adam/Variable_5
Ď
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
use_locking(*
_output_shapes	
:*
T0
ľ
training/Adam/Assign_17Assigndense_2/biastraining/Adam/sub_19*
use_locking(*
T0*
_output_shapes	
:*
validate_shape(*
_class
loc:@dense_2/bias
v
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes
:	
[
training/Adam/sub_20/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
q
training/Adam/add_19Addtraining/Adam/mul_31training/Adam/mul_32*
T0*
_output_shapes
:	
w
training/Adam/mul_33MulAdam/beta_2/readtraining/Adam/Variable_14/read*
T0*
_output_shapes
:	
[
training/Adam/sub_21/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes
:	
n
training/Adam/mul_35Multraining/Adam/multraining/Adam/add_19*
_output_shapes
:	*
T0
[
training/Adam/Const_30Const*
valueB
 *    *
_output_shapes
: *
dtype0
[
training/Adam/Const_31Const*
valueB
 *  *
dtype0*
_output_shapes
: 

%training/Adam/clip_by_value_7/MinimumMinimumtraining/Adam/add_20training/Adam/Const_31*
_output_shapes
:	*
T0
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
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
_output_shapes
:	*
T0
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
training/Adam/Assign_18Assigntraining/Adam/Variable_6training/Adam/add_19*+
_class!
loc:@training/Adam/Variable_6*
_output_shapes
:	*
use_locking(*
T0*
validate_shape(
Ó
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*
_output_shapes
:	*
validate_shape(*
T0*
use_locking(*,
_class"
 loc:@training/Adam/Variable_14
˝
training/Adam/Assign_20Assigndense_3/kerneltraining/Adam/sub_22*
T0*
_output_shapes
:	*
validate_shape(*!
_class
loc:@dense_3/kernel*
use_locking(
q
training/Adam/mul_36MulAdam/beta_1/readtraining/Adam/Variable_7/read*
T0*
_output_shapes
:
[
training/Adam/sub_23/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
f
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
_output_shapes
: *
T0
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
training/Adam/sub_24/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes
:
[
training/Adam/Const_32Const*
valueB
 *    *
_output_shapes
: *
dtype0
[
training/Adam/Const_33Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
training/Adam/add_24/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
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
training/Adam/sub_25Subdense_3/bias/readtraining/Adam/truediv_8*
_output_shapes
:*
T0
Ě
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*+
_class!
loc:@training/Adam/Variable_7*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
Î
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*
use_locking(*
T0*,
_class"
 loc:@training/Adam/Variable_15*
validate_shape(*
_output_shapes
:
´
training/Adam/Assign_23Assigndense_3/biastraining/Adam/sub_25*
_output_shapes
:*
validate_shape(*
_class
loc:@dense_3/bias*
T0*
use_locking(
ˇ
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/Adam/AssignAdd^training/Adam/Assign^training/Adam/Assign_1^training/Adam/Assign_2^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23
0

group_depsNoOp	^loss/mul^metrics/acc/Mean

IsVariableInitializedIsVariableInitializeddense/kernel*
dtype0*
_class
loc:@dense/kernel*
_output_shapes
: 

IsVariableInitialized_1IsVariableInitialized
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_2IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
dtype0

IsVariableInitialized_3IsVariableInitializeddense_1/bias*
dtype0*
_class
loc:@dense_1/bias*
_output_shapes
: 
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
: *
dtype0*!
_class
loc:@dense_3/kernel

IsVariableInitialized_7IsVariableInitializeddense_3/bias*
dtype0*
_class
loc:@dense_3/bias*
_output_shapes
: 

IsVariableInitialized_8IsVariableInitializedAdam/iterations*"
_class
loc:@Adam/iterations*
dtype0	*
_output_shapes
: 
z
IsVariableInitialized_9IsVariableInitializedAdam/lr*
dtype0*
_class
loc:@Adam/lr*
_output_shapes
: 

IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
dtype0*
_class
loc:@Adam/beta_1*
_output_shapes
: 

IsVariableInitialized_11IsVariableInitializedAdam/beta_2*
_class
loc:@Adam/beta_2*
_output_shapes
: *
dtype0

IsVariableInitialized_12IsVariableInitialized
Adam/decay*
dtype0*
_class
loc:@Adam/decay*
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
IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_1*
_output_shapes
: *
dtype0*+
_class!
loc:@training/Adam/Variable_1

IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_2*+
_class!
loc:@training/Adam/Variable_2*
_output_shapes
: *
dtype0

IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_3*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_3

IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_4

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*
_output_shapes
: *
dtype0*+
_class!
loc:@training/Adam/Variable_5
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
IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
: *
dtype0

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_9*
_output_shapes
: *
dtype0*+
_class!
loc:@training/Adam/Variable_9

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_10*
dtype0*,
_class"
 loc:@training/Adam/Variable_10*
_output_shapes
: 

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*
dtype0*,
_class"
 loc:@training/Adam/Variable_11*
_output_shapes
: 

IsVariableInitialized_25IsVariableInitializedtraining/Adam/Variable_12*,
_class"
 loc:@training/Adam/Variable_12*
dtype0*
_output_shapes
: 

IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
: *
dtype0

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_14*
dtype0*,
_class"
 loc:@training/Adam/Variable_14*
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
initNoOp^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign ^training/Adam/Variable_2/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign
p
dense_5_inputPlaceholder*
shape:˙˙˙˙˙˙˙˙˙1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
dtype0
Ł
/dense_4/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_4/kernel*
_output_shapes
:*
valueB"1      *
dtype0

-dense_4/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_4/kernel*
_output_shapes
: *
valueB
 *<ž*
dtype0

-dense_4/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *<>*!
_class
loc:@dense_4/kernel
ě
7dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_4/kernel/Initializer/random_uniform/shape*

seed *
dtype0*!
_class
loc:@dense_4/kernel*
_output_shapes
:	1*
T0*
seed2 
Ö
-dense_4/kernel/Initializer/random_uniform/subSub-dense_4/kernel/Initializer/random_uniform/max-dense_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *!
_class
loc:@dense_4/kernel*
T0
é
-dense_4/kernel/Initializer/random_uniform/mulMul7dense_4/kernel/Initializer/random_uniform/RandomUniform-dense_4/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_4/kernel*
T0*
_output_shapes
:	1
Ű
)dense_4/kernel/Initializer/random_uniformAdd-dense_4/kernel/Initializer/random_uniform/mul-dense_4/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	1*!
_class
loc:@dense_4/kernel
§
dense_4/kernel
VariableV2*
	container *
shape:	1*
_output_shapes
:	1*!
_class
loc:@dense_4/kernel*
shared_name *
dtype0
Đ
dense_4/kernel/AssignAssigndense_4/kernel)dense_4/kernel/Initializer/random_uniform*
use_locking(*!
_class
loc:@dense_4/kernel*
T0*
validate_shape(*
_output_shapes
:	1
|
dense_4/kernel/readIdentitydense_4/kernel*
_output_shapes
:	1*!
_class
loc:@dense_4/kernel*
T0

dense_4/bias/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes	
:*
_class
loc:@dense_4/bias

dense_4/bias
VariableV2*
_class
loc:@dense_4/bias*
	container *
_output_shapes	
:*
shape:*
shared_name *
dtype0
ť
dense_4/bias/AssignAssigndense_4/biasdense_4/bias/Initializer/zeros*
_class
loc:@dense_4/bias*
_output_shapes	
:*
validate_shape(*
T0*
use_locking(
r
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes	
:*
T0

dense_5/MatMulMatMuldense_5_inputdense_4/kernel/read*
transpose_b( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
T0

dense_5/BiasAddBiasAdddense_5/MatMuldense_4/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
X
dense_5/ReluReludense_5/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_5/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*!
_class
loc:@dense_5/kernel

-dense_5/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_5/kernel*
valueB
 *   ž

-dense_5/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_5/kernel*
valueB
 *   >*
_output_shapes
: *
dtype0
í
7dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_5/kernel/Initializer/random_uniform/shape*
dtype0*!
_class
loc:@dense_5/kernel*

seed *
T0* 
_output_shapes
:
*
seed2 
Ö
-dense_5/kernel/Initializer/random_uniform/subSub-dense_5/kernel/Initializer/random_uniform/max-dense_5/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_5/kernel*
T0*
_output_shapes
: 
ę
-dense_5/kernel/Initializer/random_uniform/mulMul7dense_5/kernel/Initializer/random_uniform/RandomUniform-dense_5/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*!
_class
loc:@dense_5/kernel
Ü
)dense_5/kernel/Initializer/random_uniformAdd-dense_5/kernel/Initializer/random_uniform/mul-dense_5/kernel/Initializer/random_uniform/min* 
_output_shapes
:
*!
_class
loc:@dense_5/kernel*
T0
Š
dense_5/kernel
VariableV2*
dtype0* 
_output_shapes
:
*!
_class
loc:@dense_5/kernel*
	container *
shape:
*
shared_name 
Ń
dense_5/kernel/AssignAssigndense_5/kernel)dense_5/kernel/Initializer/random_uniform*
T0*!
_class
loc:@dense_5/kernel* 
_output_shapes
:
*
use_locking(*
validate_shape(
}
dense_5/kernel/readIdentitydense_5/kernel*!
_class
loc:@dense_5/kernel*
T0* 
_output_shapes
:


dense_5/bias/Initializer/zerosConst*
valueB*    *
_output_shapes	
:*
dtype0*
_class
loc:@dense_5/bias

dense_5/bias
VariableV2*
dtype0*
shared_name *
shape:*
	container *
_class
loc:@dense_5/bias*
_output_shapes	
:
ť
dense_5/bias/AssignAssigndense_5/biasdense_5/bias/Initializer/zeros*
use_locking(*
_class
loc:@dense_5/bias*
T0*
validate_shape(*
_output_shapes	
:
r
dense_5/bias/readIdentitydense_5/bias*
_class
loc:@dense_5/bias*
T0*
_output_shapes	
:

dense_6/MatMulMatMuldense_5/Reludense_5/kernel/read*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dense_6/BiasAddBiasAdddense_6/MatMuldense_5/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
X
dense_6/ReluReludense_6/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_6/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_output_shapes
:*!
_class
loc:@dense_6/kernel*
dtype0

-dense_6/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_6/kernel*
_output_shapes
: *
valueB
 *óľ˝

-dense_6/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_6/kernel*
valueB
 *óľ=*
_output_shapes
: *
dtype0
í
7dense_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_6/kernel/Initializer/random_uniform/shape*

seed *
T0*
seed2 *!
_class
loc:@dense_6/kernel*
dtype0* 
_output_shapes
:

Ö
-dense_6/kernel/Initializer/random_uniform/subSub-dense_6/kernel/Initializer/random_uniform/max-dense_6/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *!
_class
loc:@dense_6/kernel
ę
-dense_6/kernel/Initializer/random_uniform/mulMul7dense_6/kernel/Initializer/random_uniform/RandomUniform-dense_6/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
*
T0*!
_class
loc:@dense_6/kernel
Ü
)dense_6/kernel/Initializer/random_uniformAdd-dense_6/kernel/Initializer/random_uniform/mul-dense_6/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_6/kernel*
T0* 
_output_shapes
:

Š
dense_6/kernel
VariableV2*
shared_name *
shape:
*
	container *!
_class
loc:@dense_6/kernel*
dtype0* 
_output_shapes
:

Ń
dense_6/kernel/AssignAssigndense_6/kernel)dense_6/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*!
_class
loc:@dense_6/kernel* 
_output_shapes
:

}
dense_6/kernel/readIdentitydense_6/kernel*!
_class
loc:@dense_6/kernel*
T0* 
_output_shapes
:


dense_6/bias/Initializer/zerosConst*
_output_shapes	
:*
_class
loc:@dense_6/bias*
dtype0*
valueB*    

dense_6/bias
VariableV2*
_output_shapes	
:*
shared_name *
shape:*
	container *
_class
loc:@dense_6/bias*
dtype0
ť
dense_6/bias/AssignAssigndense_6/biasdense_6/bias/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:*
_class
loc:@dense_6/bias
r
dense_6/bias/readIdentitydense_6/bias*
_class
loc:@dense_6/bias*
_output_shapes	
:*
T0

dense_7/MatMulMatMuldense_6/Reludense_6/kernel/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *
transpose_b( 

dense_7/BiasAddBiasAdddense_7/MatMuldense_6/bias/read*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC
X
dense_7/ReluReludense_7/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_7/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_7/kernel*
_output_shapes
:*
dtype0

-dense_7/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *Ű˝*!
_class
loc:@dense_7/kernel

-dense_7/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *Ű=*!
_class
loc:@dense_7/kernel*
_output_shapes
: 
ě
7dense_7/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_7/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_7/kernel*
dtype0*
_output_shapes
:	
*
seed2 *

seed *
T0
Ö
-dense_7/kernel/Initializer/random_uniform/subSub-dense_7/kernel/Initializer/random_uniform/max-dense_7/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_7/kernel
é
-dense_7/kernel/Initializer/random_uniform/mulMul7dense_7/kernel/Initializer/random_uniform/RandomUniform-dense_7/kernel/Initializer/random_uniform/sub*
_output_shapes
:	
*!
_class
loc:@dense_7/kernel*
T0
Ű
)dense_7/kernel/Initializer/random_uniformAdd-dense_7/kernel/Initializer/random_uniform/mul-dense_7/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_7/kernel*
T0*
_output_shapes
:	

§
dense_7/kernel
VariableV2*
_output_shapes
:	
*
shared_name *
dtype0*
	container *
shape:	
*!
_class
loc:@dense_7/kernel
Đ
dense_7/kernel/AssignAssigndense_7/kernel)dense_7/kernel/Initializer/random_uniform*
use_locking(*!
_class
loc:@dense_7/kernel*
validate_shape(*
T0*
_output_shapes
:	

|
dense_7/kernel/readIdentitydense_7/kernel*!
_class
loc:@dense_7/kernel*
T0*
_output_shapes
:	


dense_7/bias/Initializer/zerosConst*
_class
loc:@dense_7/bias*
valueB
*    *
dtype0*
_output_shapes
:


dense_7/bias
VariableV2*
dtype0*
_output_shapes
:
*
shape:
*
shared_name *
_class
loc:@dense_7/bias*
	container 
ş
dense_7/bias/AssignAssigndense_7/biasdense_7/bias/Initializer/zeros*
_output_shapes
:
*
_class
loc:@dense_7/bias*
validate_shape(*
T0*
use_locking(
q
dense_7/bias/readIdentitydense_7/bias*
T0*
_output_shapes
:
*
_class
loc:@dense_7/bias

dense_8/MatMulMatMuldense_7/Reludense_7/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*
transpose_b( *
transpose_a( 

dense_8/BiasAddBiasAdddense_8/MatMuldense_7/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

]
dense_8/SoftmaxSoftmaxdense_8/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

a
Adam_1/iterations/initial_valueConst*
value	B	 R *
_output_shapes
: *
dtype0	
u
Adam_1/iterations
VariableV2*
shared_name *
_output_shapes
: *
dtype0	*
	container *
shape: 
Ć
Adam_1/iterations/AssignAssignAdam_1/iterationsAdam_1/iterations/initial_value*
use_locking(*
validate_shape(*
_output_shapes
: *
T0	*$
_class
loc:@Adam_1/iterations
|
Adam_1/iterations/readIdentityAdam_1/iterations*
T0	*
_output_shapes
: *$
_class
loc:@Adam_1/iterations
\
Adam_1/lr/initial_valueConst*
_output_shapes
: *
valueB
 *ˇŃ8*
dtype0
m
	Adam_1/lr
VariableV2*
shape: *
shared_name *
_output_shapes
: *
	container *
dtype0
Ś
Adam_1/lr/AssignAssign	Adam_1/lrAdam_1/lr/initial_value*
use_locking(*
T0*
_output_shapes
: *
_class
loc:@Adam_1/lr*
validate_shape(
d
Adam_1/lr/readIdentity	Adam_1/lr*
_class
loc:@Adam_1/lr*
T0*
_output_shapes
: 
`
Adam_1/beta_1/initial_valueConst*
_output_shapes
: *
valueB
 *fff?*
dtype0
q
Adam_1/beta_1
VariableV2*
	container *
_output_shapes
: *
shape: *
shared_name *
dtype0
ś
Adam_1/beta_1/AssignAssignAdam_1/beta_1Adam_1/beta_1/initial_value*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@Adam_1/beta_1*
validate_shape(
p
Adam_1/beta_1/readIdentityAdam_1/beta_1*
_output_shapes
: * 
_class
loc:@Adam_1/beta_1*
T0
`
Adam_1/beta_2/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *wž?
q
Adam_1/beta_2
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *
shared_name 
ś
Adam_1/beta_2/AssignAssignAdam_1/beta_2Adam_1/beta_2/initial_value* 
_class
loc:@Adam_1/beta_2*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
p
Adam_1/beta_2/readIdentityAdam_1/beta_2*
T0* 
_class
loc:@Adam_1/beta_2*
_output_shapes
: 
_
Adam_1/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
p
Adam_1/decay
VariableV2*
shape: *
	container *
shared_name *
_output_shapes
: *
dtype0
˛
Adam_1/decay/AssignAssignAdam_1/decayAdam_1/decay/initial_value*
validate_shape(*
use_locking(*
_class
loc:@Adam_1/decay*
_output_shapes
: *
T0
m
Adam_1/decay/readIdentityAdam_1/decay*
T0*
_output_shapes
: *
_class
loc:@Adam_1/decay

dense_8_targetPlaceholder*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0
q
dense_8_sample_weightsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
^
loss_1/dense_8_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *żÖ3
^
loss_1/dense_8_loss/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
u
loss_1/dense_8_loss/subSubloss_1/dense_8_loss/sub/xloss_1/dense_8_loss/Const*
T0*
_output_shapes
: 

)loss_1/dense_8_loss/clip_by_value/MinimumMinimumdense_8/Softmaxloss_1/dense_8_loss/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
¤
!loss_1/dense_8_loss/clip_by_valueMaximum)loss_1/dense_8_loss/clip_by_value/Minimumloss_1/dense_8_loss/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

s
loss_1/dense_8_loss/LogLog!loss_1/dense_8_loss/clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
t
!loss_1/dense_8_loss/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙

loss_1/dense_8_loss/ReshapeReshapedense_8_target!loss_1/dense_8_loss/Reshape/shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
z
loss_1/dense_8_loss/CastCastloss_1/dense_8_loss/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0*

DstT0	
t
#loss_1/dense_8_loss/Reshape_1/shapeConst*
dtype0*
valueB"˙˙˙˙
   *
_output_shapes
:
Ś
loss_1/dense_8_loss/Reshape_1Reshapeloss_1/dense_8_loss/Log#loss_1/dense_8_loss/Reshape_1/shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


=loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss_1/dense_8_loss/Cast*
T0	*
_output_shapes
:*
out_type0

[loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss_1/dense_8_loss/Reshape_1loss_1/dense_8_loss/Cast*6
_output_shapes$
":˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*
Tlabels0	*
T0
m
*loss_1/dense_8_loss/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB 
ô
loss_1/dense_8_loss/MeanMean[loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*loss_1/dense_8_loss/Mean/reduction_indices*

Tidx0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( 
~
loss_1/dense_8_loss/mulMulloss_1/dense_8_loss/Meandense_8_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
loss_1/dense_8_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 

loss_1/dense_8_loss/NotEqualNotEqualdense_8_sample_weightsloss_1/dense_8_loss/NotEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
}
loss_1/dense_8_loss/Cast_1Castloss_1/dense_8_loss/NotEqual*

SrcT0
*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

DstT0
e
loss_1/dense_8_loss/Const_1Const*
_output_shapes
:*
valueB: *
dtype0

loss_1/dense_8_loss/Mean_1Meanloss_1/dense_8_loss/Cast_1loss_1/dense_8_loss/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0

loss_1/dense_8_loss/truedivRealDivloss_1/dense_8_loss/mulloss_1/dense_8_loss/Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
loss_1/dense_8_loss/Const_2Const*
dtype0*
valueB: *
_output_shapes
:

loss_1/dense_8_loss/Mean_2Meanloss_1/dense_8_loss/truedivloss_1/dense_8_loss/Const_2*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
Q
loss_1/mul/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
\

loss_1/mulMulloss_1/mul/xloss_1/dense_8_loss/Mean_2*
_output_shapes
: *
T0
n
#metrics_1/acc/Max/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙

metrics_1/acc/MaxMaxdense_8_target#metrics_1/acc/Max/reduction_indices*
	keep_dims( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
i
metrics_1/acc/ArgMax/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *
dtype0

metrics_1/acc/ArgMaxArgMaxdense_8/Softmaxmetrics_1/acc/ArgMax/dimension*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0*
output_type0	
m
metrics_1/acc/CastCastmetrics_1/acc/ArgMax*

SrcT0	*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
q
metrics_1/acc/EqualEqualmetrics_1/acc/Maxmetrics_1/acc/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
metrics_1/acc/Cast_1Castmetrics_1/acc/Equal*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0
*

DstT0
]
metrics_1/acc/ConstConst*
valueB: *
_output_shapes
:*
dtype0

metrics_1/acc/MeanMeanmetrics_1/acc/Cast_1metrics_1/acc/Const*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 

training_1/Adam/gradients/ShapeConst*
_class
loc:@loss_1/mul*
_output_shapes
: *
dtype0*
valueB 

#training_1/Adam/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
_class
loc:@loss_1/mul*
dtype0
Ź
training_1/Adam/gradients/FillFilltraining_1/Adam/gradients/Shape#training_1/Adam/gradients/grad_ys_0*
T0*
_class
loc:@loss_1/mul*
_output_shapes
: 
°
-training_1/Adam/gradients/loss_1/mul_grad/MulMultraining_1/Adam/gradients/Fillloss_1/dense_8_loss/Mean_2*
T0*
_class
loc:@loss_1/mul*
_output_shapes
: 
¤
/training_1/Adam/gradients/loss_1/mul_grad/Mul_1Multraining_1/Adam/gradients/Fillloss_1/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss_1/mul
Ŕ
Gtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
¨
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ReshapeReshape/training_1/Adam/gradients/loss_1/mul_grad/Mul_1Gtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Reshape/shape*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
T0*
_output_shapes
:*
Tshape0
É
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ShapeShapeloss_1/dense_8_loss/truediv*
T0*
_output_shapes
:*
out_type0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
š
>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/TileTileAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Reshape?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape*

Tmultiples0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape_1Shapeloss_1/dense_8_loss/truediv*
T0*
_output_shapes
:*
out_type0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
ł
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape_2Const*
_output_shapes
: *
valueB *-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
dtype0
¸
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ConstConst*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
_output_shapes
:*
dtype0*
valueB: 
ˇ
>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ProdProdAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape_1?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Const*
_output_shapes
: *
	keep_dims( *-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*

Tidx0*
T0
ş
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: *-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
ť
@training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Prod_1ProdAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape_2Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Const_1*
	keep_dims( *

Tidx0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
T0*
_output_shapes
: 
´
Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Maximum/yConst*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
_output_shapes
: *
dtype0*
value	B :
Ł
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/MaximumMaximum@training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Prod_1Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: *-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
Ą
Btraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/floordivFloorDiv>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ProdAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Maximum*
T0*
_output_shapes
: *-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
é
>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/CastCastBtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
Š
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/truedivRealDiv>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Tile>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
Ç
@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/ShapeShapeloss_1/dense_8_loss/mul*
T0*.
_class$
" loc:@loss_1/dense_8_loss/truediv*
_output_shapes
:*
out_type0
ľ
Btraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Shape_1Const*.
_class$
" loc:@loss_1/dense_8_loss/truediv*
_output_shapes
: *
dtype0*
valueB 
Ü
Ptraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/ShapeBtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@loss_1/dense_8_loss/truediv*
T0

Btraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDivRealDivAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/truedivloss_1/dense_8_loss/Mean_1*
T0*.
_class$
" loc:@loss_1/dense_8_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ë
>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/SumSumBtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDivPtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss_1/dense_8_loss/truediv
ť
Btraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/ReshapeReshape>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Sum@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Shape*.
_class$
" loc:@loss_1/dense_8_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
ź
>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/NegNegloss_1/dense_8_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@loss_1/dense_8_loss/truediv*
T0

Dtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDiv_1RealDiv>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Negloss_1/dense_8_loss/Mean_1*.
_class$
" loc:@loss_1/dense_8_loss/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDiv_2RealDivDtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDiv_1loss_1/dense_8_loss/Mean_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@loss_1/dense_8_loss/truediv
Ź
>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/mulMulAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/truedivDtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDiv_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@loss_1/dense_8_loss/truediv
Ë
@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Sum_1Sum>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/mulRtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/BroadcastGradientArgs:1*
T0*.
_class$
" loc:@loss_1/dense_8_loss/truediv*
	keep_dims( *
_output_shapes
:*

Tidx0
´
Dtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Reshape_1Reshape@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Sum_1Btraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Shape_1*
T0*
Tshape0*.
_class$
" loc:@loss_1/dense_8_loss/truediv*
_output_shapes
: 
Ŕ
<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/ShapeShapeloss_1/dense_8_loss/Mean*
T0*
_output_shapes
:**
_class 
loc:@loss_1/dense_8_loss/mul*
out_type0
Ŕ
>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape_1Shapedense_8_sample_weights*
T0*
_output_shapes
:*
out_type0**
_class 
loc:@loss_1/dense_8_loss/mul
Ě
Ltraining_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape_1**
_class 
loc:@loss_1/dense_8_loss/mul*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
÷
:training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/mulMulBtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Reshapedense_8_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@loss_1/dense_8_loss/mul*
T0
ˇ
:training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/SumSum:training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/mulLtraining_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( **
_class 
loc:@loss_1/dense_8_loss/mul*

Tidx0*
T0
Ť
>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/ReshapeReshape:training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Sum<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape*
Tshape0*
T0**
_class 
loc:@loss_1/dense_8_loss/mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ű
<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/mul_1Mulloss_1/dense_8_loss/MeanBtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@loss_1/dense_8_loss/mul*
T0
˝
<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Sum_1Sum<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/mul_1Ntraining_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:**
_class 
loc:@loss_1/dense_8_loss/mul*
	keep_dims( *

Tidx0
ą
@training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Reshape_1Reshape<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Sum_1>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@loss_1/dense_8_loss/mul

=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ShapeShape[loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
out_type0*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean
Ť
<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/SizeConst*
value	B :*
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean*
dtype0
ü
;training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/addAdd*loss_1/dense_8_loss/Mean/reduction_indices<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Size*
T0*
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean

;training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/modFloorMod;training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/add<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
: 
ś
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: *+
_class!
loc:@loss_1/dense_8_loss/Mean
˛
Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range/startConst*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
: *
value	B : *
dtype0
˛
Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range/deltaConst*
value	B :*+
_class!
loc:@loss_1/dense_8_loss/Mean*
dtype0*
_output_shapes
: 
ă
=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/rangeRangeCtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range/start<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/SizeCtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range/delta*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
:*

Tidx0
ą
Btraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Fill/valueConst*
dtype0*
value	B :*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
: 

<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/FillFill?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_1Btraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Fill/value*
T0*
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean
ł
Etraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/DynamicStitchDynamicStitch=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range;training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/mod=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Fill*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*+
_class!
loc:@loss_1/dense_8_loss/Mean*
N*
T0
°
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0*+
_class!
loc:@loss_1/dense_8_loss/Mean
Ż
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/MaximumMaximumEtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/DynamicStitchAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean
§
@training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/floordivFloorDiv=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*+
_class!
loc:@loss_1/dense_8_loss/Mean*
T0
Ż
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ReshapeReshape>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/ReshapeEtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:*+
_class!
loc:@loss_1/dense_8_loss/Mean
Š
<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/TileTile?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Reshape@training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/floordiv*
_output_shapes
:*+
_class!
loc:@loss_1/dense_8_loss/Mean*
T0*

Tmultiples0

?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_2Shape[loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean*
out_type0
Ä
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_3Shapeloss_1/dense_8_loss/Mean*
_output_shapes
:*
out_type0*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean
´
=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ConstConst*+
_class!
loc:@loss_1/dense_8_loss/Mean*
dtype0*
_output_shapes
:*
valueB: 
Ż
<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ProdProd?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_2=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Const*

Tidx0*
	keep_dims( *+
_class!
loc:@loss_1/dense_8_loss/Mean*
T0*
_output_shapes
: 
ś
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Const_1Const*
_output_shapes
:*+
_class!
loc:@loss_1/dense_8_loss/Mean*
dtype0*
valueB: 
ł
>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Prod_1Prod?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_3?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Const_1*

Tidx0*+
_class!
loc:@loss_1/dense_8_loss/Mean*
T0*
_output_shapes
: *
	keep_dims( 
˛
Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :*+
_class!
loc:@loss_1/dense_8_loss/Mean

Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum_1Maximum>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Prod_1Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum_1/y*+
_class!
loc:@loss_1/dense_8_loss/Mean*
T0*
_output_shapes
: 

Btraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/floordiv_1FloorDiv<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ProdAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum_1*
T0*
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean
ĺ
<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/CastCastBtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/floordiv_1*

SrcT0*

DstT0*
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean
Ą
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/truedivRealDiv<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Tile<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Cast*+
_class!
loc:@loss_1/dense_8_loss/Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
˛
$training_1/Adam/gradients/zeros_like	ZerosLike]loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
Ö
training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient]loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*´
message¨ĽCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ĺ
training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
dtype0

training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/truedivtraining_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
~training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMultraining_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimstraining_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ë
Btraining_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/ShapeShapeloss_1/dense_8_loss/Log*0
_class&
$"loc:@loss_1/dense_8_loss/Reshape_1*
out_type0*
T0*
_output_shapes
:

Dtraining_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/ReshapeReshape~training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulBtraining_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*0
_class&
$"loc:@loss_1/dense_8_loss/Reshape_1*
Tshape0*
T0

Atraining_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/Reciprocal
Reciprocal!loss_1/dense_8_loss/clip_by_valueE^training_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_class 
loc:@loss_1/dense_8_loss/Log*
T0
¨
:training_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/mulMulDtraining_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/ReshapeAtraining_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_class 
loc:@loss_1/dense_8_loss/Log
ĺ
Ftraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/ShapeShape)loss_1/dense_8_loss/clip_by_value/Minimum*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
_output_shapes
:*
out_type0*
T0
Á
Htraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB *4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value
ř
Htraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_2Shape:training_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/mul*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
_output_shapes
:*
T0*
out_type0
Ç
Ltraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zeros/ConstConst*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
dtype0*
valueB
 *    *
_output_shapes
: 
Î
Ftraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zerosFillHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_2Ltraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zeros/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
T0

Mtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/GreaterEqualGreaterEqual)loss_1/dense_8_loss/clip_by_value/Minimumloss_1/dense_8_loss/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value
ô
Vtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/ShapeHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value

Gtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/SelectSelectMtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/GreaterEqual:training_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/mulFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zeros*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value

Itraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Select_1SelectMtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/GreaterEqualFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zeros:training_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value
â
Dtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/SumSumGtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/SelectVtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/BroadcastGradientArgs*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
×
Htraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/ReshapeReshapeDtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/SumFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value
č
Ftraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Sum_1SumItraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Select_1Xtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/BroadcastGradientArgs:1*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
Ě
Jtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Reshape_1ReshapeFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Sum_1Htraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: *4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value
Ű
Ntraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/ShapeShapedense_8/Softmax*
T0*
_output_shapes
:*
out_type0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum
Ń
Ptraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_1Const*
valueB *<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
_output_shapes
: *
dtype0

Ptraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_2ShapeHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Reshape*
out_type0*
_output_shapes
:*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
T0
×
Ttraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum
î
Ntraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zerosFillPtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_2Ttraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zeros/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
T0
ů
Rtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualdense_8/Softmaxloss_1/dense_8_loss/sub*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


^training_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/ShapePtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_1*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ˇ
Otraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/SelectSelectRtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/LessEqualHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/ReshapeNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zeros*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

š
Qtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Select_1SelectRtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/LessEqualNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zerosHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Reshape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum

Ltraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/SumSumOtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Select^training_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
	keep_dims( 
÷
Ptraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/ReshapeReshapeLtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/SumNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
Tshape0*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum

Ntraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Sum_1SumQtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Select_1`training_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*

Tidx0*
T0
ě
Rtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Sum_1Ptraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_1*
Tshape0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
_output_shapes
: *
T0
ň
2training_1/Adam/gradients/dense_8/Softmax_grad/mulMulPtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Reshapedense_8/Softmax*"
_class
loc:@dense_8/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
˛
Dtraining_1/Adam/gradients/dense_8/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0*"
_class
loc:@dense_8/Softmax
˘
2training_1/Adam/gradients/dense_8/Softmax_grad/SumSum2training_1/Adam/gradients/dense_8/Softmax_grad/mulDtraining_1/Adam/gradients/dense_8/Softmax_grad/Sum/reduction_indices*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*"
_class
loc:@dense_8/Softmax*
	keep_dims( 
ą
<training_1/Adam/gradients/dense_8/Softmax_grad/Reshape/shapeConst*
_output_shapes
:*"
_class
loc:@dense_8/Softmax*
dtype0*
valueB"˙˙˙˙   

6training_1/Adam/gradients/dense_8/Softmax_grad/ReshapeReshape2training_1/Adam/gradients/dense_8/Softmax_grad/Sum<training_1/Adam/gradients/dense_8/Softmax_grad/Reshape/shape*"
_class
loc:@dense_8/Softmax*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0

2training_1/Adam/gradients/dense_8/Softmax_grad/subSubPtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Reshape6training_1/Adam/gradients/dense_8/Softmax_grad/Reshape*"
_class
loc:@dense_8/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
Ö
4training_1/Adam/gradients/dense_8/Softmax_grad/mul_1Mul2training_1/Adam/gradients/dense_8/Softmax_grad/subdense_8/Softmax*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*"
_class
loc:@dense_8/Softmax
ß
:training_1/Adam/gradients/dense_8/BiasAdd_grad/BiasAddGradBiasAddGrad4training_1/Adam/gradients/dense_8/Softmax_grad/mul_1*
data_formatNHWC*"
_class
loc:@dense_8/BiasAdd*
T0*
_output_shapes
:


4training_1/Adam/gradients/dense_8/MatMul_grad/MatMulMatMul4training_1/Adam/gradients/dense_8/Softmax_grad/mul_1dense_7/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( *!
_class
loc:@dense_8/MatMul*
transpose_b(*
T0
÷
6training_1/Adam/gradients/dense_8/MatMul_grad/MatMul_1MatMuldense_7/Relu4training_1/Adam/gradients/dense_8/Softmax_grad/mul_1*
transpose_b( *
transpose_a(*
T0*!
_class
loc:@dense_8/MatMul*
_output_shapes
:	

Ř
4training_1/Adam/gradients/dense_7/Relu_grad/ReluGradReluGrad4training_1/Adam/gradients/dense_8/MatMul_grad/MatMuldense_7/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@dense_7/Relu
ŕ
:training_1/Adam/gradients/dense_7/BiasAdd_grad/BiasAddGradBiasAddGrad4training_1/Adam/gradients/dense_7/Relu_grad/ReluGrad*
data_formatNHWC*
T0*"
_class
loc:@dense_7/BiasAdd*
_output_shapes	
:

4training_1/Adam/gradients/dense_7/MatMul_grad/MatMulMatMul4training_1/Adam/gradients/dense_7/Relu_grad/ReluGraddense_6/kernel/read*
T0*
transpose_a( *
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*!
_class
loc:@dense_7/MatMul
ř
6training_1/Adam/gradients/dense_7/MatMul_grad/MatMul_1MatMuldense_6/Relu4training_1/Adam/gradients/dense_7/Relu_grad/ReluGrad*
T0*
transpose_a(* 
_output_shapes
:
*!
_class
loc:@dense_7/MatMul*
transpose_b( 
Ř
4training_1/Adam/gradients/dense_6/Relu_grad/ReluGradReluGrad4training_1/Adam/gradients/dense_7/MatMul_grad/MatMuldense_6/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@dense_6/Relu*
T0
ŕ
:training_1/Adam/gradients/dense_6/BiasAdd_grad/BiasAddGradBiasAddGrad4training_1/Adam/gradients/dense_6/Relu_grad/ReluGrad*
_output_shapes	
:*
data_formatNHWC*
T0*"
_class
loc:@dense_6/BiasAdd

4training_1/Adam/gradients/dense_6/MatMul_grad/MatMulMatMul4training_1/Adam/gradients/dense_6/Relu_grad/ReluGraddense_5/kernel/read*
T0*
transpose_a( *!
_class
loc:@dense_6/MatMul*
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ř
6training_1/Adam/gradients/dense_6/MatMul_grad/MatMul_1MatMuldense_5/Relu4training_1/Adam/gradients/dense_6/Relu_grad/ReluGrad* 
_output_shapes
:
*
transpose_b( *
transpose_a(*
T0*!
_class
loc:@dense_6/MatMul
Ř
4training_1/Adam/gradients/dense_5/Relu_grad/ReluGradReluGrad4training_1/Adam/gradients/dense_6/MatMul_grad/MatMuldense_5/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@dense_5/Relu
ŕ
:training_1/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGradBiasAddGrad4training_1/Adam/gradients/dense_5/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*"
_class
loc:@dense_5/BiasAdd*
data_formatNHWC

4training_1/Adam/gradients/dense_5/MatMul_grad/MatMulMatMul4training_1/Adam/gradients/dense_5/Relu_grad/ReluGraddense_4/kernel/read*!
_class
loc:@dense_5/MatMul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
transpose_a( *
transpose_b(
ř
6training_1/Adam/gradients/dense_5/MatMul_grad/MatMul_1MatMuldense_5_input4training_1/Adam/gradients/dense_5/Relu_grad/ReluGrad*
transpose_a(*
transpose_b( *!
_class
loc:@dense_5/MatMul*
T0*
_output_shapes
:	1
a
training_1/Adam/AssignAdd/valueConst*
value	B	 R*
_output_shapes
: *
dtype0	
´
training_1/Adam/AssignAdd	AssignAddAdam_1/iterationstraining_1/Adam/AssignAdd/value*
T0	*
_output_shapes
: *
use_locking( *$
_class
loc:@Adam_1/iterations
d
training_1/Adam/CastCastAdam_1/iterations/read*
_output_shapes
: *

SrcT0	*

DstT0
Z
training_1/Adam/add/yConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
h
training_1/Adam/addAddtraining_1/Adam/Casttraining_1/Adam/add/y*
T0*
_output_shapes
: 
d
training_1/Adam/PowPowAdam_1/beta_2/readtraining_1/Adam/add*
_output_shapes
: *
T0
Z
training_1/Adam/sub/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
g
training_1/Adam/subSubtraining_1/Adam/sub/xtraining_1/Adam/Pow*
T0*
_output_shapes
: 
Z
training_1/Adam/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
training_1/Adam/Const_1Const*
dtype0*
valueB
 *  *
_output_shapes
: 

%training_1/Adam/clip_by_value/MinimumMinimumtraining_1/Adam/subtraining_1/Adam/Const_1*
_output_shapes
: *
T0

training_1/Adam/clip_by_valueMaximum%training_1/Adam/clip_by_value/Minimumtraining_1/Adam/Const*
T0*
_output_shapes
: 
\
training_1/Adam/SqrtSqrttraining_1/Adam/clip_by_value*
_output_shapes
: *
T0
f
training_1/Adam/Pow_1PowAdam_1/beta_1/readtraining_1/Adam/add*
T0*
_output_shapes
: 
\
training_1/Adam/sub_1/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
m
training_1/Adam/sub_1Subtraining_1/Adam/sub_1/xtraining_1/Adam/Pow_1*
_output_shapes
: *
T0
p
training_1/Adam/truedivRealDivtraining_1/Adam/Sqrttraining_1/Adam/sub_1*
T0*
_output_shapes
: 
d
training_1/Adam/mulMulAdam_1/lr/readtraining_1/Adam/truediv*
T0*
_output_shapes
: 
n
training_1/Adam/Const_2Const*
_output_shapes
:	1*
valueB	1*    *
dtype0

training_1/Adam/Variable
VariableV2*
shape:	1*
	container *
shared_name *
dtype0*
_output_shapes
:	1
Ü
training_1/Adam/Variable/AssignAssigntraining_1/Adam/Variabletraining_1/Adam/Const_2*
validate_shape(*
_output_shapes
:	1*
T0*+
_class!
loc:@training_1/Adam/Variable*
use_locking(

training_1/Adam/Variable/readIdentitytraining_1/Adam/Variable*
_output_shapes
:	1*+
_class!
loc:@training_1/Adam/Variable*
T0
f
training_1/Adam/Const_3Const*
dtype0*
valueB*    *
_output_shapes	
:

training_1/Adam/Variable_1
VariableV2*
dtype0*
shared_name *
	container *
shape:*
_output_shapes	
:
Ţ
!training_1/Adam/Variable_1/AssignAssigntraining_1/Adam/Variable_1training_1/Adam/Const_3*
use_locking(*
T0*
validate_shape(*-
_class#
!loc:@training_1/Adam/Variable_1*
_output_shapes	
:

training_1/Adam/Variable_1/readIdentitytraining_1/Adam/Variable_1*
_output_shapes	
:*
T0*-
_class#
!loc:@training_1/Adam/Variable_1
p
training_1/Adam/Const_4Const*
valueB
*    *
dtype0* 
_output_shapes
:


training_1/Adam/Variable_2
VariableV2*
shape:
*
	container * 
_output_shapes
:
*
dtype0*
shared_name 
ă
!training_1/Adam/Variable_2/AssignAssigntraining_1/Adam/Variable_2training_1/Adam/Const_4*
use_locking(*
validate_shape(*
T0*-
_class#
!loc:@training_1/Adam/Variable_2* 
_output_shapes
:

Ą
training_1/Adam/Variable_2/readIdentitytraining_1/Adam/Variable_2* 
_output_shapes
:
*
T0*-
_class#
!loc:@training_1/Adam/Variable_2
f
training_1/Adam/Const_5Const*
valueB*    *
dtype0*
_output_shapes	
:

training_1/Adam/Variable_3
VariableV2*
shape:*
shared_name *
	container *
dtype0*
_output_shapes	
:
Ţ
!training_1/Adam/Variable_3/AssignAssigntraining_1/Adam/Variable_3training_1/Adam/Const_5*
use_locking(*
_output_shapes	
:*-
_class#
!loc:@training_1/Adam/Variable_3*
T0*
validate_shape(

training_1/Adam/Variable_3/readIdentitytraining_1/Adam/Variable_3*-
_class#
!loc:@training_1/Adam/Variable_3*
_output_shapes	
:*
T0
p
training_1/Adam/Const_6Const* 
_output_shapes
:
*
valueB
*    *
dtype0

training_1/Adam/Variable_4
VariableV2*
dtype0*
shape:
*
shared_name *
	container * 
_output_shapes
:

ă
!training_1/Adam/Variable_4/AssignAssigntraining_1/Adam/Variable_4training_1/Adam/Const_6*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*-
_class#
!loc:@training_1/Adam/Variable_4
Ą
training_1/Adam/Variable_4/readIdentitytraining_1/Adam/Variable_4*
T0*-
_class#
!loc:@training_1/Adam/Variable_4* 
_output_shapes
:

f
training_1/Adam/Const_7Const*
_output_shapes	
:*
dtype0*
valueB*    

training_1/Adam/Variable_5
VariableV2*
	container *
dtype0*
shape:*
_output_shapes	
:*
shared_name 
Ţ
!training_1/Adam/Variable_5/AssignAssigntraining_1/Adam/Variable_5training_1/Adam/Const_7*
use_locking(*-
_class#
!loc:@training_1/Adam/Variable_5*
validate_shape(*
T0*
_output_shapes	
:

training_1/Adam/Variable_5/readIdentitytraining_1/Adam/Variable_5*
T0*
_output_shapes	
:*-
_class#
!loc:@training_1/Adam/Variable_5
n
training_1/Adam/Const_8Const*
_output_shapes
:	
*
dtype0*
valueB	
*    

training_1/Adam/Variable_6
VariableV2*
shared_name *
shape:	
*
dtype0*
_output_shapes
:	
*
	container 
â
!training_1/Adam/Variable_6/AssignAssigntraining_1/Adam/Variable_6training_1/Adam/Const_8*
validate_shape(*
_output_shapes
:	
*-
_class#
!loc:@training_1/Adam/Variable_6*
T0*
use_locking(
 
training_1/Adam/Variable_6/readIdentitytraining_1/Adam/Variable_6*
_output_shapes
:	
*-
_class#
!loc:@training_1/Adam/Variable_6*
T0
d
training_1/Adam/Const_9Const*
dtype0*
_output_shapes
:
*
valueB
*    

training_1/Adam/Variable_7
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
:
*
shape:

Ý
!training_1/Adam/Variable_7/AssignAssigntraining_1/Adam/Variable_7training_1/Adam/Const_9*
validate_shape(*
use_locking(*-
_class#
!loc:@training_1/Adam/Variable_7*
T0*
_output_shapes
:


training_1/Adam/Variable_7/readIdentitytraining_1/Adam/Variable_7*
_output_shapes
:
*-
_class#
!loc:@training_1/Adam/Variable_7*
T0
o
training_1/Adam/Const_10Const*
_output_shapes
:	1*
dtype0*
valueB	1*    

training_1/Adam/Variable_8
VariableV2*
dtype0*
	container *
_output_shapes
:	1*
shape:	1*
shared_name 
ă
!training_1/Adam/Variable_8/AssignAssigntraining_1/Adam/Variable_8training_1/Adam/Const_10*
use_locking(*-
_class#
!loc:@training_1/Adam/Variable_8*
T0*
validate_shape(*
_output_shapes
:	1
 
training_1/Adam/Variable_8/readIdentitytraining_1/Adam/Variable_8*
_output_shapes
:	1*
T0*-
_class#
!loc:@training_1/Adam/Variable_8
g
training_1/Adam/Const_11Const*
dtype0*
_output_shapes	
:*
valueB*    

training_1/Adam/Variable_9
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shared_name *
shape:
ß
!training_1/Adam/Variable_9/AssignAssigntraining_1/Adam/Variable_9training_1/Adam/Const_11*-
_class#
!loc:@training_1/Adam/Variable_9*
_output_shapes	
:*
validate_shape(*
use_locking(*
T0

training_1/Adam/Variable_9/readIdentitytraining_1/Adam/Variable_9*-
_class#
!loc:@training_1/Adam/Variable_9*
T0*
_output_shapes	
:
q
training_1/Adam/Const_12Const* 
_output_shapes
:
*
valueB
*    *
dtype0

training_1/Adam/Variable_10
VariableV2*
	container * 
_output_shapes
:
*
dtype0*
shared_name *
shape:

ç
"training_1/Adam/Variable_10/AssignAssigntraining_1/Adam/Variable_10training_1/Adam/Const_12*
validate_shape(*.
_class$
" loc:@training_1/Adam/Variable_10*
use_locking(* 
_output_shapes
:
*
T0
¤
 training_1/Adam/Variable_10/readIdentitytraining_1/Adam/Variable_10* 
_output_shapes
:
*
T0*.
_class$
" loc:@training_1/Adam/Variable_10
g
training_1/Adam/Const_13Const*
_output_shapes	
:*
dtype0*
valueB*    

training_1/Adam/Variable_11
VariableV2*
_output_shapes	
:*
shape:*
dtype0*
	container *
shared_name 
â
"training_1/Adam/Variable_11/AssignAssigntraining_1/Adam/Variable_11training_1/Adam/Const_13*.
_class$
" loc:@training_1/Adam/Variable_11*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:

 training_1/Adam/Variable_11/readIdentitytraining_1/Adam/Variable_11*
_output_shapes	
:*
T0*.
_class$
" loc:@training_1/Adam/Variable_11
q
training_1/Adam/Const_14Const*
dtype0* 
_output_shapes
:
*
valueB
*    

training_1/Adam/Variable_12
VariableV2*
dtype0*
shape:
* 
_output_shapes
:
*
shared_name *
	container 
ç
"training_1/Adam/Variable_12/AssignAssigntraining_1/Adam/Variable_12training_1/Adam/Const_14*
validate_shape(*
T0*.
_class$
" loc:@training_1/Adam/Variable_12* 
_output_shapes
:
*
use_locking(
¤
 training_1/Adam/Variable_12/readIdentitytraining_1/Adam/Variable_12*
T0* 
_output_shapes
:
*.
_class$
" loc:@training_1/Adam/Variable_12
g
training_1/Adam/Const_15Const*
valueB*    *
_output_shapes	
:*
dtype0

training_1/Adam/Variable_13
VariableV2*
_output_shapes	
:*
shape:*
dtype0*
	container *
shared_name 
â
"training_1/Adam/Variable_13/AssignAssigntraining_1/Adam/Variable_13training_1/Adam/Const_15*
use_locking(*
_output_shapes	
:*.
_class$
" loc:@training_1/Adam/Variable_13*
T0*
validate_shape(

 training_1/Adam/Variable_13/readIdentitytraining_1/Adam/Variable_13*
T0*.
_class$
" loc:@training_1/Adam/Variable_13*
_output_shapes	
:
o
training_1/Adam/Const_16Const*
_output_shapes
:	
*
valueB	
*    *
dtype0

training_1/Adam/Variable_14
VariableV2*
	container *
shape:	
*
shared_name *
_output_shapes
:	
*
dtype0
ć
"training_1/Adam/Variable_14/AssignAssigntraining_1/Adam/Variable_14training_1/Adam/Const_16*.
_class$
" loc:@training_1/Adam/Variable_14*
_output_shapes
:	
*
T0*
use_locking(*
validate_shape(
Ł
 training_1/Adam/Variable_14/readIdentitytraining_1/Adam/Variable_14*.
_class$
" loc:@training_1/Adam/Variable_14*
T0*
_output_shapes
:	

e
training_1/Adam/Const_17Const*
valueB
*    *
dtype0*
_output_shapes
:


training_1/Adam/Variable_15
VariableV2*
shared_name *
shape:
*
	container *
dtype0*
_output_shapes
:

á
"training_1/Adam/Variable_15/AssignAssigntraining_1/Adam/Variable_15training_1/Adam/Const_17*
_output_shapes
:
*.
_class$
" loc:@training_1/Adam/Variable_15*
use_locking(*
validate_shape(*
T0

 training_1/Adam/Variable_15/readIdentitytraining_1/Adam/Variable_15*.
_class$
" loc:@training_1/Adam/Variable_15*
T0*
_output_shapes
:

y
training_1/Adam/mul_1MulAdam_1/beta_1/readtraining_1/Adam/Variable/read*
_output_shapes
:	1*
T0
\
training_1/Adam/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
j
training_1/Adam/sub_2Subtraining_1/Adam/sub_2/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_2Multraining_1/Adam/sub_26training_1/Adam/gradients/dense_5/MatMul_grad/MatMul_1*
_output_shapes
:	1*
T0
t
training_1/Adam/add_1Addtraining_1/Adam/mul_1training_1/Adam/mul_2*
_output_shapes
:	1*
T0
{
training_1/Adam/mul_3MulAdam_1/beta_2/readtraining_1/Adam/Variable_8/read*
T0*
_output_shapes
:	1
\
training_1/Adam/sub_3/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
j
training_1/Adam/sub_3Subtraining_1/Adam/sub_3/xAdam_1/beta_2/read*
_output_shapes
: *
T0

training_1/Adam/SquareSquare6training_1/Adam/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	1
u
training_1/Adam/mul_4Multraining_1/Adam/sub_3training_1/Adam/Square*
_output_shapes
:	1*
T0
t
training_1/Adam/add_2Addtraining_1/Adam/mul_3training_1/Adam/mul_4*
_output_shapes
:	1*
T0
r
training_1/Adam/mul_5Multraining_1/Adam/multraining_1/Adam/add_1*
T0*
_output_shapes
:	1
]
training_1/Adam/Const_18Const*
dtype0*
_output_shapes
: *
valueB
 *    
]
training_1/Adam/Const_19Const*
valueB
 *  *
dtype0*
_output_shapes
: 

'training_1/Adam/clip_by_value_1/MinimumMinimumtraining_1/Adam/add_2training_1/Adam/Const_19*
T0*
_output_shapes
:	1

training_1/Adam/clip_by_value_1Maximum'training_1/Adam/clip_by_value_1/Minimumtraining_1/Adam/Const_18*
_output_shapes
:	1*
T0
i
training_1/Adam/Sqrt_1Sqrttraining_1/Adam/clip_by_value_1*
_output_shapes
:	1*
T0
\
training_1/Adam/add_3/yConst*
dtype0*
valueB
 *wĚ+2*
_output_shapes
: 
w
training_1/Adam/add_3Addtraining_1/Adam/Sqrt_1training_1/Adam/add_3/y*
_output_shapes
:	1*
T0
|
training_1/Adam/truediv_1RealDivtraining_1/Adam/mul_5training_1/Adam/add_3*
T0*
_output_shapes
:	1
v
training_1/Adam/sub_4Subdense_4/kernel/readtraining_1/Adam/truediv_1*
T0*
_output_shapes
:	1
Ń
training_1/Adam/AssignAssigntraining_1/Adam/Variabletraining_1/Adam/add_1*
_output_shapes
:	1*+
_class!
loc:@training_1/Adam/Variable*
T0*
validate_shape(*
use_locking(
×
training_1/Adam/Assign_1Assigntraining_1/Adam/Variable_8training_1/Adam/add_2*-
_class#
!loc:@training_1/Adam/Variable_8*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	1
ż
training_1/Adam/Assign_2Assigndense_4/kerneltraining_1/Adam/sub_4*
T0*
_output_shapes
:	1*
validate_shape(*!
_class
loc:@dense_4/kernel*
use_locking(
w
training_1/Adam/mul_6MulAdam_1/beta_1/readtraining_1/Adam/Variable_1/read*
T0*
_output_shapes	
:
\
training_1/Adam/sub_5/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
j
training_1/Adam/sub_5Subtraining_1/Adam/sub_5/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_7Multraining_1/Adam/sub_5:training_1/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
p
training_1/Adam/add_4Addtraining_1/Adam/mul_6training_1/Adam/mul_7*
_output_shapes	
:*
T0
w
training_1/Adam/mul_8MulAdam_1/beta_2/readtraining_1/Adam/Variable_9/read*
_output_shapes	
:*
T0
\
training_1/Adam/sub_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
j
training_1/Adam/sub_6Subtraining_1/Adam/sub_6/xAdam_1/beta_2/read*
T0*
_output_shapes
: 

training_1/Adam/Square_1Square:training_1/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
s
training_1/Adam/mul_9Multraining_1/Adam/sub_6training_1/Adam/Square_1*
_output_shapes	
:*
T0
p
training_1/Adam/add_5Addtraining_1/Adam/mul_8training_1/Adam/mul_9*
T0*
_output_shapes	
:
o
training_1/Adam/mul_10Multraining_1/Adam/multraining_1/Adam/add_4*
_output_shapes	
:*
T0
]
training_1/Adam/Const_20Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training_1/Adam/Const_21Const*
dtype0*
_output_shapes
: *
valueB
 *  

'training_1/Adam/clip_by_value_2/MinimumMinimumtraining_1/Adam/add_5training_1/Adam/Const_21*
_output_shapes	
:*
T0

training_1/Adam/clip_by_value_2Maximum'training_1/Adam/clip_by_value_2/Minimumtraining_1/Adam/Const_20*
T0*
_output_shapes	
:
e
training_1/Adam/Sqrt_2Sqrttraining_1/Adam/clip_by_value_2*
T0*
_output_shapes	
:
\
training_1/Adam/add_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
s
training_1/Adam/add_6Addtraining_1/Adam/Sqrt_2training_1/Adam/add_6/y*
_output_shapes	
:*
T0
y
training_1/Adam/truediv_2RealDivtraining_1/Adam/mul_10training_1/Adam/add_6*
_output_shapes	
:*
T0
p
training_1/Adam/sub_7Subdense_4/bias/readtraining_1/Adam/truediv_2*
T0*
_output_shapes	
:
Ó
training_1/Adam/Assign_3Assigntraining_1/Adam/Variable_1training_1/Adam/add_4*
_output_shapes	
:*
use_locking(*
T0*-
_class#
!loc:@training_1/Adam/Variable_1*
validate_shape(
Ó
training_1/Adam/Assign_4Assigntraining_1/Adam/Variable_9training_1/Adam/add_5*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(*-
_class#
!loc:@training_1/Adam/Variable_9
ˇ
training_1/Adam/Assign_5Assigndense_4/biastraining_1/Adam/sub_7*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:*
_class
loc:@dense_4/bias
}
training_1/Adam/mul_11MulAdam_1/beta_1/readtraining_1/Adam/Variable_2/read* 
_output_shapes
:
*
T0
\
training_1/Adam/sub_8/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
j
training_1/Adam/sub_8Subtraining_1/Adam/sub_8/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_12Multraining_1/Adam/sub_86training_1/Adam/gradients/dense_6/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
w
training_1/Adam/add_7Addtraining_1/Adam/mul_11training_1/Adam/mul_12* 
_output_shapes
:
*
T0
~
training_1/Adam/mul_13MulAdam_1/beta_2/read training_1/Adam/Variable_10/read*
T0* 
_output_shapes
:

\
training_1/Adam/sub_9/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
j
training_1/Adam/sub_9Subtraining_1/Adam/sub_9/xAdam_1/beta_2/read*
T0*
_output_shapes
: 

training_1/Adam/Square_2Square6training_1/Adam/gradients/dense_6/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

y
training_1/Adam/mul_14Multraining_1/Adam/sub_9training_1/Adam/Square_2*
T0* 
_output_shapes
:

w
training_1/Adam/add_8Addtraining_1/Adam/mul_13training_1/Adam/mul_14*
T0* 
_output_shapes
:

t
training_1/Adam/mul_15Multraining_1/Adam/multraining_1/Adam/add_7*
T0* 
_output_shapes
:

]
training_1/Adam/Const_22Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training_1/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 

'training_1/Adam/clip_by_value_3/MinimumMinimumtraining_1/Adam/add_8training_1/Adam/Const_23*
T0* 
_output_shapes
:


training_1/Adam/clip_by_value_3Maximum'training_1/Adam/clip_by_value_3/Minimumtraining_1/Adam/Const_22*
T0* 
_output_shapes
:

j
training_1/Adam/Sqrt_3Sqrttraining_1/Adam/clip_by_value_3*
T0* 
_output_shapes
:

\
training_1/Adam/add_9/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
x
training_1/Adam/add_9Addtraining_1/Adam/Sqrt_3training_1/Adam/add_9/y*
T0* 
_output_shapes
:

~
training_1/Adam/truediv_3RealDivtraining_1/Adam/mul_15training_1/Adam/add_9*
T0* 
_output_shapes
:

x
training_1/Adam/sub_10Subdense_5/kernel/readtraining_1/Adam/truediv_3*
T0* 
_output_shapes
:

Ř
training_1/Adam/Assign_6Assigntraining_1/Adam/Variable_2training_1/Adam/add_7*-
_class#
!loc:@training_1/Adam/Variable_2* 
_output_shapes
:
*
use_locking(*
T0*
validate_shape(
Ú
training_1/Adam/Assign_7Assigntraining_1/Adam/Variable_10training_1/Adam/add_8*
validate_shape(*.
_class$
" loc:@training_1/Adam/Variable_10* 
_output_shapes
:
*
T0*
use_locking(
Á
training_1/Adam/Assign_8Assigndense_5/kerneltraining_1/Adam/sub_10*
use_locking(* 
_output_shapes
:
*
T0*
validate_shape(*!
_class
loc:@dense_5/kernel
x
training_1/Adam/mul_16MulAdam_1/beta_1/readtraining_1/Adam/Variable_3/read*
T0*
_output_shapes	
:
]
training_1/Adam/sub_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
training_1/Adam/sub_11Subtraining_1/Adam/sub_11/xAdam_1/beta_1/read*
T0*
_output_shapes
: 

training_1/Adam/mul_17Multraining_1/Adam/sub_11:training_1/Adam/gradients/dense_6/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
s
training_1/Adam/add_10Addtraining_1/Adam/mul_16training_1/Adam/mul_17*
_output_shapes	
:*
T0
y
training_1/Adam/mul_18MulAdam_1/beta_2/read training_1/Adam/Variable_11/read*
_output_shapes	
:*
T0
]
training_1/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
l
training_1/Adam/sub_12Subtraining_1/Adam/sub_12/xAdam_1/beta_2/read*
T0*
_output_shapes
: 

training_1/Adam/Square_3Square:training_1/Adam/gradients/dense_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
u
training_1/Adam/mul_19Multraining_1/Adam/sub_12training_1/Adam/Square_3*
_output_shapes	
:*
T0
s
training_1/Adam/add_11Addtraining_1/Adam/mul_18training_1/Adam/mul_19*
T0*
_output_shapes	
:
p
training_1/Adam/mul_20Multraining_1/Adam/multraining_1/Adam/add_10*
_output_shapes	
:*
T0
]
training_1/Adam/Const_24Const*
valueB
 *    *
_output_shapes
: *
dtype0
]
training_1/Adam/Const_25Const*
dtype0*
_output_shapes
: *
valueB
 *  

'training_1/Adam/clip_by_value_4/MinimumMinimumtraining_1/Adam/add_11training_1/Adam/Const_25*
_output_shapes	
:*
T0

training_1/Adam/clip_by_value_4Maximum'training_1/Adam/clip_by_value_4/Minimumtraining_1/Adam/Const_24*
_output_shapes	
:*
T0
e
training_1/Adam/Sqrt_4Sqrttraining_1/Adam/clip_by_value_4*
_output_shapes	
:*
T0
]
training_1/Adam/add_12/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
u
training_1/Adam/add_12Addtraining_1/Adam/Sqrt_4training_1/Adam/add_12/y*
_output_shapes	
:*
T0
z
training_1/Adam/truediv_4RealDivtraining_1/Adam/mul_20training_1/Adam/add_12*
_output_shapes	
:*
T0
q
training_1/Adam/sub_13Subdense_5/bias/readtraining_1/Adam/truediv_4*
_output_shapes	
:*
T0
Ô
training_1/Adam/Assign_9Assigntraining_1/Adam/Variable_3training_1/Adam/add_10*
T0*-
_class#
!loc:@training_1/Adam/Variable_3*
validate_shape(*
_output_shapes	
:*
use_locking(
×
training_1/Adam/Assign_10Assigntraining_1/Adam/Variable_11training_1/Adam/add_11*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(*.
_class$
" loc:@training_1/Adam/Variable_11
š
training_1/Adam/Assign_11Assigndense_5/biastraining_1/Adam/sub_13*
use_locking(*
validate_shape(*
T0*
_class
loc:@dense_5/bias*
_output_shapes	
:
}
training_1/Adam/mul_21MulAdam_1/beta_1/readtraining_1/Adam/Variable_4/read* 
_output_shapes
:
*
T0
]
training_1/Adam/sub_14/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
training_1/Adam/sub_14Subtraining_1/Adam/sub_14/xAdam_1/beta_1/read*
T0*
_output_shapes
: 

training_1/Adam/mul_22Multraining_1/Adam/sub_146training_1/Adam/gradients/dense_7/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
x
training_1/Adam/add_13Addtraining_1/Adam/mul_21training_1/Adam/mul_22*
T0* 
_output_shapes
:

~
training_1/Adam/mul_23MulAdam_1/beta_2/read training_1/Adam/Variable_12/read* 
_output_shapes
:
*
T0
]
training_1/Adam/sub_15/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
l
training_1/Adam/sub_15Subtraining_1/Adam/sub_15/xAdam_1/beta_2/read*
_output_shapes
: *
T0

training_1/Adam/Square_4Square6training_1/Adam/gradients/dense_7/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
z
training_1/Adam/mul_24Multraining_1/Adam/sub_15training_1/Adam/Square_4* 
_output_shapes
:
*
T0
x
training_1/Adam/add_14Addtraining_1/Adam/mul_23training_1/Adam/mul_24* 
_output_shapes
:
*
T0
u
training_1/Adam/mul_25Multraining_1/Adam/multraining_1/Adam/add_13*
T0* 
_output_shapes
:

]
training_1/Adam/Const_26Const*
valueB
 *    *
_output_shapes
: *
dtype0
]
training_1/Adam/Const_27Const*
dtype0*
_output_shapes
: *
valueB
 *  

'training_1/Adam/clip_by_value_5/MinimumMinimumtraining_1/Adam/add_14training_1/Adam/Const_27* 
_output_shapes
:
*
T0

training_1/Adam/clip_by_value_5Maximum'training_1/Adam/clip_by_value_5/Minimumtraining_1/Adam/Const_26* 
_output_shapes
:
*
T0
j
training_1/Adam/Sqrt_5Sqrttraining_1/Adam/clip_by_value_5* 
_output_shapes
:
*
T0
]
training_1/Adam/add_15/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
z
training_1/Adam/add_15Addtraining_1/Adam/Sqrt_5training_1/Adam/add_15/y* 
_output_shapes
:
*
T0

training_1/Adam/truediv_5RealDivtraining_1/Adam/mul_25training_1/Adam/add_15*
T0* 
_output_shapes
:

x
training_1/Adam/sub_16Subdense_6/kernel/readtraining_1/Adam/truediv_5*
T0* 
_output_shapes
:

Ú
training_1/Adam/Assign_12Assigntraining_1/Adam/Variable_4training_1/Adam/add_13*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(*-
_class#
!loc:@training_1/Adam/Variable_4
Ü
training_1/Adam/Assign_13Assigntraining_1/Adam/Variable_12training_1/Adam/add_14*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(*.
_class$
" loc:@training_1/Adam/Variable_12
Â
training_1/Adam/Assign_14Assigndense_6/kerneltraining_1/Adam/sub_16*
use_locking(*
validate_shape(*
T0*!
_class
loc:@dense_6/kernel* 
_output_shapes
:

x
training_1/Adam/mul_26MulAdam_1/beta_1/readtraining_1/Adam/Variable_5/read*
T0*
_output_shapes	
:
]
training_1/Adam/sub_17/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training_1/Adam/sub_17Subtraining_1/Adam/sub_17/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_27Multraining_1/Adam/sub_17:training_1/Adam/gradients/dense_7/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
s
training_1/Adam/add_16Addtraining_1/Adam/mul_26training_1/Adam/mul_27*
_output_shapes	
:*
T0
y
training_1/Adam/mul_28MulAdam_1/beta_2/read training_1/Adam/Variable_13/read*
_output_shapes	
:*
T0
]
training_1/Adam/sub_18/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
l
training_1/Adam/sub_18Subtraining_1/Adam/sub_18/xAdam_1/beta_2/read*
T0*
_output_shapes
: 

training_1/Adam/Square_5Square:training_1/Adam/gradients/dense_7/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
u
training_1/Adam/mul_29Multraining_1/Adam/sub_18training_1/Adam/Square_5*
T0*
_output_shapes	
:
s
training_1/Adam/add_17Addtraining_1/Adam/mul_28training_1/Adam/mul_29*
T0*
_output_shapes	
:
p
training_1/Adam/mul_30Multraining_1/Adam/multraining_1/Adam/add_16*
T0*
_output_shapes	
:
]
training_1/Adam/Const_28Const*
dtype0*
valueB
 *    *
_output_shapes
: 
]
training_1/Adam/Const_29Const*
valueB
 *  *
_output_shapes
: *
dtype0

'training_1/Adam/clip_by_value_6/MinimumMinimumtraining_1/Adam/add_17training_1/Adam/Const_29*
_output_shapes	
:*
T0

training_1/Adam/clip_by_value_6Maximum'training_1/Adam/clip_by_value_6/Minimumtraining_1/Adam/Const_28*
T0*
_output_shapes	
:
e
training_1/Adam/Sqrt_6Sqrttraining_1/Adam/clip_by_value_6*
_output_shapes	
:*
T0
]
training_1/Adam/add_18/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
u
training_1/Adam/add_18Addtraining_1/Adam/Sqrt_6training_1/Adam/add_18/y*
_output_shapes	
:*
T0
z
training_1/Adam/truediv_6RealDivtraining_1/Adam/mul_30training_1/Adam/add_18*
T0*
_output_shapes	
:
q
training_1/Adam/sub_19Subdense_6/bias/readtraining_1/Adam/truediv_6*
_output_shapes	
:*
T0
Ő
training_1/Adam/Assign_15Assigntraining_1/Adam/Variable_5training_1/Adam/add_16*
T0*
use_locking(*
_output_shapes	
:*-
_class#
!loc:@training_1/Adam/Variable_5*
validate_shape(
×
training_1/Adam/Assign_16Assigntraining_1/Adam/Variable_13training_1/Adam/add_17*
T0*
validate_shape(*.
_class$
" loc:@training_1/Adam/Variable_13*
_output_shapes	
:*
use_locking(
š
training_1/Adam/Assign_17Assigndense_6/biastraining_1/Adam/sub_19*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:*
_class
loc:@dense_6/bias
|
training_1/Adam/mul_31MulAdam_1/beta_1/readtraining_1/Adam/Variable_6/read*
T0*
_output_shapes
:	

]
training_1/Adam/sub_20/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
training_1/Adam/sub_20Subtraining_1/Adam/sub_20/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_32Multraining_1/Adam/sub_206training_1/Adam/gradients/dense_8/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
w
training_1/Adam/add_19Addtraining_1/Adam/mul_31training_1/Adam/mul_32*
_output_shapes
:	
*
T0
}
training_1/Adam/mul_33MulAdam_1/beta_2/read training_1/Adam/Variable_14/read*
T0*
_output_shapes
:	

]
training_1/Adam/sub_21/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training_1/Adam/sub_21Subtraining_1/Adam/sub_21/xAdam_1/beta_2/read*
_output_shapes
: *
T0

training_1/Adam/Square_6Square6training_1/Adam/gradients/dense_8/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	

y
training_1/Adam/mul_34Multraining_1/Adam/sub_21training_1/Adam/Square_6*
_output_shapes
:	
*
T0
w
training_1/Adam/add_20Addtraining_1/Adam/mul_33training_1/Adam/mul_34*
T0*
_output_shapes
:	

t
training_1/Adam/mul_35Multraining_1/Adam/multraining_1/Adam/add_19*
T0*
_output_shapes
:	

]
training_1/Adam/Const_30Const*
_output_shapes
: *
dtype0*
valueB
 *    
]
training_1/Adam/Const_31Const*
_output_shapes
: *
valueB
 *  *
dtype0

'training_1/Adam/clip_by_value_7/MinimumMinimumtraining_1/Adam/add_20training_1/Adam/Const_31*
_output_shapes
:	
*
T0

training_1/Adam/clip_by_value_7Maximum'training_1/Adam/clip_by_value_7/Minimumtraining_1/Adam/Const_30*
_output_shapes
:	
*
T0
i
training_1/Adam/Sqrt_7Sqrttraining_1/Adam/clip_by_value_7*
_output_shapes
:	
*
T0
]
training_1/Adam/add_21/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
y
training_1/Adam/add_21Addtraining_1/Adam/Sqrt_7training_1/Adam/add_21/y*
_output_shapes
:	
*
T0
~
training_1/Adam/truediv_7RealDivtraining_1/Adam/mul_35training_1/Adam/add_21*
T0*
_output_shapes
:	

w
training_1/Adam/sub_22Subdense_7/kernel/readtraining_1/Adam/truediv_7*
_output_shapes
:	
*
T0
Ů
training_1/Adam/Assign_18Assigntraining_1/Adam/Variable_6training_1/Adam/add_19*-
_class#
!loc:@training_1/Adam/Variable_6*
_output_shapes
:	
*
use_locking(*
T0*
validate_shape(
Ű
training_1/Adam/Assign_19Assigntraining_1/Adam/Variable_14training_1/Adam/add_20*.
_class$
" loc:@training_1/Adam/Variable_14*
T0*
_output_shapes
:	
*
use_locking(*
validate_shape(
Á
training_1/Adam/Assign_20Assigndense_7/kerneltraining_1/Adam/sub_22*
validate_shape(*!
_class
loc:@dense_7/kernel*
use_locking(*
_output_shapes
:	
*
T0
w
training_1/Adam/mul_36MulAdam_1/beta_1/readtraining_1/Adam/Variable_7/read*
T0*
_output_shapes
:

]
training_1/Adam/sub_23/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
training_1/Adam/sub_23Subtraining_1/Adam/sub_23/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_37Multraining_1/Adam/sub_23:training_1/Adam/gradients/dense_8/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
r
training_1/Adam/add_22Addtraining_1/Adam/mul_36training_1/Adam/mul_37*
_output_shapes
:
*
T0
x
training_1/Adam/mul_38MulAdam_1/beta_2/read training_1/Adam/Variable_15/read*
T0*
_output_shapes
:

]
training_1/Adam/sub_24/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training_1/Adam/sub_24Subtraining_1/Adam/sub_24/xAdam_1/beta_2/read*
_output_shapes
: *
T0

training_1/Adam/Square_7Square:training_1/Adam/gradients/dense_8/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:

t
training_1/Adam/mul_39Multraining_1/Adam/sub_24training_1/Adam/Square_7*
T0*
_output_shapes
:

r
training_1/Adam/add_23Addtraining_1/Adam/mul_38training_1/Adam/mul_39*
_output_shapes
:
*
T0
o
training_1/Adam/mul_40Multraining_1/Adam/multraining_1/Adam/add_22*
_output_shapes
:
*
T0
]
training_1/Adam/Const_32Const*
dtype0*
valueB
 *    *
_output_shapes
: 
]
training_1/Adam/Const_33Const*
dtype0*
_output_shapes
: *
valueB
 *  

'training_1/Adam/clip_by_value_8/MinimumMinimumtraining_1/Adam/add_23training_1/Adam/Const_33*
T0*
_output_shapes
:


training_1/Adam/clip_by_value_8Maximum'training_1/Adam/clip_by_value_8/Minimumtraining_1/Adam/Const_32*
_output_shapes
:
*
T0
d
training_1/Adam/Sqrt_8Sqrttraining_1/Adam/clip_by_value_8*
T0*
_output_shapes
:

]
training_1/Adam/add_24/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
t
training_1/Adam/add_24Addtraining_1/Adam/Sqrt_8training_1/Adam/add_24/y*
T0*
_output_shapes
:

y
training_1/Adam/truediv_8RealDivtraining_1/Adam/mul_40training_1/Adam/add_24*
T0*
_output_shapes
:

p
training_1/Adam/sub_25Subdense_7/bias/readtraining_1/Adam/truediv_8*
T0*
_output_shapes
:

Ô
training_1/Adam/Assign_21Assigntraining_1/Adam/Variable_7training_1/Adam/add_22*
use_locking(*-
_class#
!loc:@training_1/Adam/Variable_7*
_output_shapes
:
*
validate_shape(*
T0
Ö
training_1/Adam/Assign_22Assigntraining_1/Adam/Variable_15training_1/Adam/add_23*
use_locking(*
T0*.
_class$
" loc:@training_1/Adam/Variable_15*
validate_shape(*
_output_shapes
:

¸
training_1/Adam/Assign_23Assigndense_7/biastraining_1/Adam/sub_25*
T0*
validate_shape(*
use_locking(*
_class
loc:@dense_7/bias*
_output_shapes
:

ď
training_1/group_depsNoOp^loss_1/mul^metrics_1/acc/Mean^training_1/Adam/AssignAdd^training_1/Adam/Assign^training_1/Adam/Assign_1^training_1/Adam/Assign_2^training_1/Adam/Assign_3^training_1/Adam/Assign_4^training_1/Adam/Assign_5^training_1/Adam/Assign_6^training_1/Adam/Assign_7^training_1/Adam/Assign_8^training_1/Adam/Assign_9^training_1/Adam/Assign_10^training_1/Adam/Assign_11^training_1/Adam/Assign_12^training_1/Adam/Assign_13^training_1/Adam/Assign_14^training_1/Adam/Assign_15^training_1/Adam/Assign_16^training_1/Adam/Assign_17^training_1/Adam/Assign_18^training_1/Adam/Assign_19^training_1/Adam/Assign_20^training_1/Adam/Assign_21^training_1/Adam/Assign_22^training_1/Adam/Assign_23
6
group_deps_1NoOp^loss_1/mul^metrics_1/acc/Mean

IsVariableInitialized_29IsVariableInitializeddense_4/kernel*
_output_shapes
: *!
_class
loc:@dense_4/kernel*
dtype0

IsVariableInitialized_30IsVariableInitializeddense_4/bias*
dtype0*
_output_shapes
: *
_class
loc:@dense_4/bias

IsVariableInitialized_31IsVariableInitializeddense_5/kernel*
dtype0*!
_class
loc:@dense_5/kernel*
_output_shapes
: 

IsVariableInitialized_32IsVariableInitializeddense_5/bias*
_class
loc:@dense_5/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_33IsVariableInitializeddense_6/kernel*!
_class
loc:@dense_6/kernel*
dtype0*
_output_shapes
: 

IsVariableInitialized_34IsVariableInitializeddense_6/bias*
_class
loc:@dense_6/bias*
dtype0*
_output_shapes
: 

IsVariableInitialized_35IsVariableInitializeddense_7/kernel*!
_class
loc:@dense_7/kernel*
_output_shapes
: *
dtype0

IsVariableInitialized_36IsVariableInitializeddense_7/bias*
dtype0*
_class
loc:@dense_7/bias*
_output_shapes
: 

IsVariableInitialized_37IsVariableInitializedAdam_1/iterations*$
_class
loc:@Adam_1/iterations*
_output_shapes
: *
dtype0	

IsVariableInitialized_38IsVariableInitialized	Adam_1/lr*
_output_shapes
: *
_class
loc:@Adam_1/lr*
dtype0

IsVariableInitialized_39IsVariableInitializedAdam_1/beta_1*
_output_shapes
: * 
_class
loc:@Adam_1/beta_1*
dtype0

IsVariableInitialized_40IsVariableInitializedAdam_1/beta_2*
_output_shapes
: * 
_class
loc:@Adam_1/beta_2*
dtype0

IsVariableInitialized_41IsVariableInitializedAdam_1/decay*
_class
loc:@Adam_1/decay*
dtype0*
_output_shapes
: 

IsVariableInitialized_42IsVariableInitializedtraining_1/Adam/Variable*
dtype0*+
_class!
loc:@training_1/Adam/Variable*
_output_shapes
: 
Ą
IsVariableInitialized_43IsVariableInitializedtraining_1/Adam/Variable_1*
dtype0*
_output_shapes
: *-
_class#
!loc:@training_1/Adam/Variable_1
Ą
IsVariableInitialized_44IsVariableInitializedtraining_1/Adam/Variable_2*-
_class#
!loc:@training_1/Adam/Variable_2*
_output_shapes
: *
dtype0
Ą
IsVariableInitialized_45IsVariableInitializedtraining_1/Adam/Variable_3*
_output_shapes
: *-
_class#
!loc:@training_1/Adam/Variable_3*
dtype0
Ą
IsVariableInitialized_46IsVariableInitializedtraining_1/Adam/Variable_4*
dtype0*-
_class#
!loc:@training_1/Adam/Variable_4*
_output_shapes
: 
Ą
IsVariableInitialized_47IsVariableInitializedtraining_1/Adam/Variable_5*
dtype0*-
_class#
!loc:@training_1/Adam/Variable_5*
_output_shapes
: 
Ą
IsVariableInitialized_48IsVariableInitializedtraining_1/Adam/Variable_6*
_output_shapes
: *-
_class#
!loc:@training_1/Adam/Variable_6*
dtype0
Ą
IsVariableInitialized_49IsVariableInitializedtraining_1/Adam/Variable_7*
_output_shapes
: *
dtype0*-
_class#
!loc:@training_1/Adam/Variable_7
Ą
IsVariableInitialized_50IsVariableInitializedtraining_1/Adam/Variable_8*
_output_shapes
: *-
_class#
!loc:@training_1/Adam/Variable_8*
dtype0
Ą
IsVariableInitialized_51IsVariableInitializedtraining_1/Adam/Variable_9*-
_class#
!loc:@training_1/Adam/Variable_9*
_output_shapes
: *
dtype0
Ł
IsVariableInitialized_52IsVariableInitializedtraining_1/Adam/Variable_10*
_output_shapes
: *.
_class$
" loc:@training_1/Adam/Variable_10*
dtype0
Ł
IsVariableInitialized_53IsVariableInitializedtraining_1/Adam/Variable_11*
dtype0*.
_class$
" loc:@training_1/Adam/Variable_11*
_output_shapes
: 
Ł
IsVariableInitialized_54IsVariableInitializedtraining_1/Adam/Variable_12*.
_class$
" loc:@training_1/Adam/Variable_12*
dtype0*
_output_shapes
: 
Ł
IsVariableInitialized_55IsVariableInitializedtraining_1/Adam/Variable_13*
_output_shapes
: *.
_class$
" loc:@training_1/Adam/Variable_13*
dtype0
Ł
IsVariableInitialized_56IsVariableInitializedtraining_1/Adam/Variable_14*
_output_shapes
: *.
_class$
" loc:@training_1/Adam/Variable_14*
dtype0
Ł
IsVariableInitialized_57IsVariableInitializedtraining_1/Adam/Variable_15*.
_class$
" loc:@training_1/Adam/Variable_15*
dtype0*
_output_shapes
: 
ü
init_1NoOp^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^dense_6/kernel/Assign^dense_6/bias/Assign^dense_7/kernel/Assign^dense_7/bias/Assign^Adam_1/iterations/Assign^Adam_1/lr/Assign^Adam_1/beta_1/Assign^Adam_1/beta_2/Assign^Adam_1/decay/Assign ^training_1/Adam/Variable/Assign"^training_1/Adam/Variable_1/Assign"^training_1/Adam/Variable_2/Assign"^training_1/Adam/Variable_3/Assign"^training_1/Adam/Variable_4/Assign"^training_1/Adam/Variable_5/Assign"^training_1/Adam/Variable_6/Assign"^training_1/Adam/Variable_7/Assign"^training_1/Adam/Variable_8/Assign"^training_1/Adam/Variable_9/Assign#^training_1/Adam/Variable_10/Assign#^training_1/Adam/Variable_11/Assign#^training_1/Adam/Variable_12/Assign#^training_1/Adam/Variable_13/Assign#^training_1/Adam/Variable_14/Assign#^training_1/Adam/Variable_15/Assign"ŢąąŘň*     ŻHšĚ	ů2Ń×AJĺŐ
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
Ttype*1.5.02v1.5.0-0-g37aa430d84ŃÍ
p
dense_1_inputPlaceholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
dtype0*
shape:˙˙˙˙˙˙˙˙˙1

-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
valueB"1      *
dtype0*
_output_shapes
:
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
+dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *<>
ć
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_output_shapes
:	1*
dtype0*

seed *
_class
loc:@dense/kernel*
seed2 *
T0
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
: 
á
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	1*
T0*
_class
loc:@dense/kernel
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	1*
_class
loc:@dense/kernel*
T0
Ł
dense/kernel
VariableV2*
_output_shapes
:	1*
dtype0*
shape:	1*
shared_name *
_class
loc:@dense/kernel*
	container 
Č
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
_output_shapes
:	1*
_class
loc:@dense/kernel*
validate_shape(*
T0*
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
valueB*    *
_class
loc:@dense/bias*
_output_shapes	
:*
dtype0


dense/bias
VariableV2*
_output_shapes	
:*
shared_name *
dtype0*
	container *
shape:*
_class
loc:@dense/bias
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
_output_shapes	
:*
validate_shape(*
_class
loc:@dense/bias*
T0*
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
transpose_b( *
T0*
transpose_a( 

dense/BiasAddBiasAdddense/MatMuldense/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
T

dense/ReluReludense/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"      *!
_class
loc:@dense_1/kernel*
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *   ž*!
_class
loc:@dense_1/kernel*
_output_shapes
: *
dtype0

-dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *   >
í
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_1/kernel*
T0*
seed2 *
dtype0*

seed * 
_output_shapes
:

Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
: 
ę
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
T0
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
*
shared_name *
shape:
*
	container *
dtype0
Ń
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
*
T0*
use_locking(*!
_class
loc:@dense_1/kernel
}
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
T0

dense_1/bias/Initializer/zerosConst*
_class
loc:@dense_1/bias*
_output_shapes	
:*
valueB*    *
dtype0

dense_1/bias
VariableV2*
_class
loc:@dense_1/bias*
_output_shapes	
:*
shape:*
	container *
dtype0*
shared_name 
ť
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
_output_shapes	
:*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0
r
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes	
:*
T0

dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
dense_2/ReluReludense_2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0*!
_class
loc:@dense_2/kernel

-dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *óľ˝*
_output_shapes
: *!
_class
loc:@dense_2/kernel*
dtype0

-dense_2/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_2/kernel*
_output_shapes
: *
valueB
 *óľ=*
dtype0
í
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_2/kernel*

seed *
dtype0*
seed2 * 
_output_shapes
:
*
T0
Ö
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes
: 
ę
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_2/kernel* 
_output_shapes
:

Ü
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
*
T0
Š
dense_2/kernel
VariableV2*
shared_name * 
_output_shapes
:
*!
_class
loc:@dense_2/kernel*
shape:
*
dtype0*
	container 
Ń
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
T0*!
_class
loc:@dense_2/kernel*
use_locking(* 
_output_shapes
:
*
validate_shape(
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
dtype0*
_class
loc:@dense_2/bias*
_output_shapes	
:*
valueB*    

dense_2/bias
VariableV2*
shape:*
_class
loc:@dense_2/bias*
dtype0*
shared_name *
_output_shapes	
:*
	container 
ť
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
use_locking(*
_class
loc:@dense_2/bias*
T0*
_output_shapes	
:*
validate_shape(
r
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
T0*
_output_shapes	
:

dense_3/MatMulMatMuldense_2/Reludense_2/kernel/read*
transpose_b( *
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

dense_3/BiasAddBiasAdddense_3/MatMuldense_2/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
dense_3/ReluReludense_3/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_3/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_3/kernel*
_output_shapes
:*
valueB"      *
dtype0

-dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *!
_class
loc:@dense_3/kernel*
valueB
 *żđÚ˝
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
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape*
T0*

seed *
_output_shapes
:	*!
_class
loc:@dense_3/kernel*
dtype0*
seed2 
Ö
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes
: 
é
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	*!
_class
loc:@dense_3/kernel
Ű
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*!
_class
loc:@dense_3/kernel
§
dense_3/kernel
VariableV2*
_output_shapes
:	*
shared_name *
	container *
shape:	*
dtype0*!
_class
loc:@dense_3/kernel
Đ
dense_3/kernel/AssignAssigndense_3/kernel)dense_3/kernel/Initializer/random_uniform*
T0*!
_class
loc:@dense_3/kernel*
use_locking(*
_output_shapes
:	*
validate_shape(
|
dense_3/kernel/readIdentitydense_3/kernel*
T0*!
_class
loc:@dense_3/kernel*
_output_shapes
:	

dense_3/bias/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *
_class
loc:@dense_3/bias

dense_3/bias
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_class
loc:@dense_3/bias*
_output_shapes
:
ş
dense_3/bias/AssignAssigndense_3/biasdense_3/bias/Initializer/zeros*
T0*
_class
loc:@dense_3/bias*
use_locking(*
validate_shape(*
_output_shapes
:
q
dense_3/bias/readIdentitydense_3/bias*
_output_shapes
:*
T0*
_class
loc:@dense_3/bias

dense_4/MatMulMatMuldense_3/Reludense_3/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
transpose_a( *
T0

dense_4/BiasAddBiasAdddense_4/MatMuldense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
dense_4/SoftmaxSoftmaxdense_4/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
Adam/iterations/initial_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
s
Adam/iterations
VariableV2*
shape: *
dtype0	*
shared_name *
	container *
_output_shapes
: 
ž
Adam/iterations/AssignAssignAdam/iterationsAdam/iterations/initial_value*
_output_shapes
: *
T0	*
use_locking(*"
_class
loc:@Adam/iterations*
validate_shape(
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
 *ˇŃ8*
dtype0*
_output_shapes
: 
k
Adam/lr
VariableV2*
	container *
shared_name *
shape: *
_output_shapes
: *
dtype0

Adam/lr/AssignAssignAdam/lrAdam/lr/initial_value*
_class
loc:@Adam/lr*
validate_shape(*
T0*
_output_shapes
: *
use_locking(
^
Adam/lr/readIdentityAdam/lr*
_class
loc:@Adam/lr*
T0*
_output_shapes
: 
^
Adam/beta_1/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?
o
Adam/beta_1
VariableV2*
_output_shapes
: *
	container *
shared_name *
shape: *
dtype0
Ž
Adam/beta_1/AssignAssignAdam/beta_1Adam/beta_1/initial_value*
_output_shapes
: *
validate_shape(*
use_locking(*
_class
loc:@Adam/beta_1*
T0
j
Adam/beta_1/readIdentityAdam/beta_1*
_output_shapes
: *
_class
loc:@Adam/beta_1*
T0
^
Adam/beta_2/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wž?
o
Adam/beta_2
VariableV2*
shared_name *
_output_shapes
: *
	container *
shape: *
dtype0
Ž
Adam/beta_2/AssignAssignAdam/beta_2Adam/beta_2/initial_value*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*
_class
loc:@Adam/beta_2
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
VariableV2*
dtype0*
	container *
_output_shapes
: *
shared_name *
shape: 
Ş
Adam/decay/AssignAssign
Adam/decayAdam/decay/initial_value*
validate_shape(*
T0*
_class
loc:@Adam/decay*
use_locking(*
_output_shapes
: 
g
Adam/decay/readIdentity
Adam/decay*
T0*
_output_shapes
: *
_class
loc:@Adam/decay

dense_4_targetPlaceholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
q
dense_4_sample_weightsPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
\
loss/dense_4_loss/ConstConst*
_output_shapes
: *
valueB
 *żÖ3*
dtype0
\
loss/dense_4_loss/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
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
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0

loss/dense_4_loss/ReshapeReshapedense_4_targetloss/dense_4_loss/Reshape/shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
v
loss/dense_4_loss/CastCastloss/dense_4_loss/Reshape*

DstT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0
r
!loss/dense_4_loss/Reshape_1/shapeConst*
valueB"˙˙˙˙   *
_output_shapes
:*
dtype0
 
loss/dense_4_loss/Reshape_1Reshapeloss/dense_4_loss/Log!loss/dense_4_loss/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0

;loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/dense_4_loss/Cast*
T0	*
out_type0*
_output_shapes
:

Yloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss/dense_4_loss/Reshape_1loss/dense_4_loss/Cast*
T0*6
_output_shapes$
":˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
Tlabels0	
k
(loss/dense_4_loss/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB 
î
loss/dense_4_loss/MeanMeanYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(loss/dense_4_loss/Mean/reduction_indices*

Tidx0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
	keep_dims( 
z
loss/dense_4_loss/mulMulloss/dense_4_loss/Meandense_4_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
a
loss/dense_4_loss/NotEqual/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 

loss/dense_4_loss/NotEqualNotEqualdense_4_sample_weightsloss/dense_4_loss/NotEqual/y*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
y
loss/dense_4_loss/Cast_1Castloss/dense_4_loss/NotEqual*

SrcT0
*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
loss/dense_4_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss/dense_4_loss/Mean_1Meanloss/dense_4_loss/Cast_1loss/dense_4_loss/Const_1*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
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
loss/dense_4_loss/Mean_2Meanloss/dense_4_loss/truedivloss/dense_4_loss/Const_2*
T0*

Tidx0*
	keep_dims( *
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
!metrics/acc/Max/reduction_indicesConst*
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: 

metrics/acc/MaxMaxdense_4_target!metrics/acc/Max/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0*
	keep_dims( 
g
metrics/acc/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

metrics/acc/ArgMaxArgMaxdense_4/Softmaxmetrics/acc/ArgMax/dimension*

Tidx0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
i
metrics/acc/CastCastmetrics/acc/ArgMax*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0	*

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

SrcT0
*

DstT0
[
metrics/acc/ConstConst*
dtype0*
valueB: *
_output_shapes
:
}
metrics/acc/MeanMeanmetrics/acc/Cast_1metrics/acc/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
}
training/Adam/gradients/ShapeConst*
_output_shapes
: *
_class
loc:@loss/mul*
dtype0*
valueB 

!training/Adam/gradients/grad_ys_0Const*
_output_shapes
: *
_class
loc:@loss/mul*
dtype0*
valueB
 *  ?
¤
training/Adam/gradients/FillFilltraining/Adam/gradients/Shape!training/Adam/gradients/grad_ys_0*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
Ś
)training/Adam/gradients/loss/mul_grad/MulMultraining/Adam/gradients/Fillloss/dense_4_loss/Mean_2*
_class
loc:@loss/mul*
_output_shapes
: *
T0

+training/Adam/gradients/loss/mul_grad/Mul_1Multraining/Adam/gradients/Fill
loss/mul/x*
_class
loc:@loss/mul*
_output_shapes
: *
T0
ş
Ctraining/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_2

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ReshapeReshape+training/Adam/gradients/loss/mul_grad/Mul_1Ctraining/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape/shape*
_output_shapes
:*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_2*
Tshape0
Á
;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ShapeShapeloss/dense_4_loss/truediv*
T0*
out_type0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_2
Ť
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/TileTile=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Reshape;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape*+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tmultiples0
Ă
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_1Shapeloss/dense_4_loss/truediv*
T0*
out_type0*
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_2
­
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_2Const*
_output_shapes
: *
dtype0*+
_class!
loc:@loss/dense_4_loss/Mean_2*
valueB 
˛
;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ConstConst*+
_class!
loc:@loss/dense_4_loss/Mean_2*
valueB: *
_output_shapes
:*
dtype0
Š
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/ProdProd=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_1;training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const*
T0*

Tidx0*+
_class!
loc:@loss/dense_4_loss/Mean_2*
_output_shapes
: *
	keep_dims( 
´
=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const_1Const*
valueB: *
_output_shapes
:*+
_class!
loc:@loss/dense_4_loss/Mean_2*
dtype0
­
<training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod_1Prod=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Shape_2=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Const_1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_2
Ž
?training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum/yConst*
value	B :*+
_class!
loc:@loss/dense_4_loss/Mean_2*
dtype0*
_output_shapes
: 

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/MaximumMaximum<training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum/y*+
_class!
loc:@loss/dense_4_loss/Mean_2*
T0*
_output_shapes
: 

>training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/floordivFloorDiv:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Maximum*
T0*
_output_shapes
: *+
_class!
loc:@loss/dense_4_loss/Mean_2
ß
:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/floordiv*

DstT0*
_output_shapes
: *

SrcT0*+
_class!
loc:@loss/dense_4_loss/Mean_2

=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truedivRealDiv:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Tile:training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/Cast*
T0*+
_class!
loc:@loss/dense_4_loss/Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/ShapeShapeloss/dense_4_loss/mul*
T0*
out_type0*
_output_shapes
:*,
_class"
 loc:@loss/dense_4_loss/truediv
Ż
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1Const*
valueB *,
_class"
 loc:@loss/dense_4_loss/truediv*
dtype0*
_output_shapes
: 
Î
Ltraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0
ţ
>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivRealDiv=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truedivloss/dense_4_loss/Mean_1*,
_class"
 loc:@loss/dense_4_loss/truediv*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
˝
:training/Adam/gradients/loss/dense_4_loss/truediv_grad/SumSum>training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDivLtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*,
_class"
 loc:@loss/dense_4_loss/truediv*
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
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2RealDiv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_1loss/dense_4_loss/Mean_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv

:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulMul=training/Adam/gradients/loss/dense_4_loss/Mean_2_grad/truediv@training/Adam/gradients/loss/dense_4_loss/truediv_grad/RealDiv_2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*,
_class"
 loc:@loss/dense_4_loss/truediv
˝
<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1Sum:training/Adam/gradients/loss/dense_4_loss/truediv_grad/mulNtraining/Adam/gradients/loss/dense_4_loss/truediv_grad/BroadcastGradientArgs:1*,
_class"
 loc:@loss/dense_4_loss/truediv*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ś
@training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape_1Reshape<training/Adam/gradients/loss/dense_4_loss/truediv_grad/Sum_1>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Shape_1*
T0*
Tshape0*,
_class"
 loc:@loss/dense_4_loss/truediv*
_output_shapes
: 
¸
8training/Adam/gradients/loss/dense_4_loss/mul_grad/ShapeShapeloss/dense_4_loss/Mean*
_output_shapes
:*(
_class
loc:@loss/dense_4_loss/mul*
T0*
out_type0
ş
:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1Shapedense_4_sample_weights*
out_type0*
T0*(
_class
loc:@loss/dense_4_loss/mul*
_output_shapes
:
ž
Htraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*(
_class
loc:@loss/dense_4_loss/mul*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
í
6training/Adam/gradients/loss/dense_4_loss/mul_grad/mulMul>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshapedense_4_sample_weights*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul*
T0
Š
6training/Adam/gradients/loss/dense_4_loss/mul_grad/SumSum6training/Adam/gradients/loss/dense_4_loss/mul_grad/mulHtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0*(
_class
loc:@loss/dense_4_loss/mul

:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeReshape6training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ď
8training/Adam/gradients/loss/dense_4_loss/mul_grad/mul_1Mulloss/dense_4_loss/Mean>training/Adam/gradients/loss/dense_4_loss/truediv_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul*
T0
Ż
8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1Sum8training/Adam/gradients/loss/dense_4_loss/mul_grad/mul_1Jtraining/Adam/gradients/loss/dense_4_loss/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( *(
_class
loc:@loss/dense_4_loss/mul
Ł
<training/Adam/gradients/loss/dense_4_loss/mul_grad/Reshape_1Reshape8training/Adam/gradients/loss/dense_4_loss/mul_grad/Sum_1:training/Adam/gradients/loss/dense_4_loss/mul_grad/Shape_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_class
loc:@loss/dense_4_loss/mul*
Tshape0
ý
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ShapeShapeYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
_output_shapes
:*
out_type0
Ľ
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/SizeConst*
dtype0*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
_output_shapes
: 
đ
7training/Adam/gradients/loss/dense_4_loss/Mean_grad/addAdd(loss/dense_4_loss/Mean/reduction_indices8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
_output_shapes
: 

7training/Adam/gradients/loss/dense_4_loss/Mean_grad/modFloorMod7training/Adam/gradients/loss/dense_4_loss/Mean_grad/add8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
T0
°
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1Const*
valueB: *)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:*
dtype0
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/startConst*
value	B : *
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/deltaConst*
value	B :*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean*
dtype0
Ń
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/rangeRange?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/start8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Size?training/Adam/gradients/loss/dense_4_loss/Mean_grad/range/delta*)
_class
loc:@loss/dense_4_loss/Mean*

Tidx0*
_output_shapes
:
Ť
>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/valueConst*
_output_shapes
: *
value	B :*)
_class
loc:@loss/dense_4_loss/Mean*
dtype0

8training/Adam/gradients/loss/dense_4_loss/Mean_grad/FillFill;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_1>training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill/value*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: 

Atraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitchDynamicStitch9training/Adam/gradients/loss/dense_4_loss/Mean_grad/range7training/Adam/gradients/loss/dense_4_loss/Mean_grad/mod9training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Fill*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :*)
_class
loc:@loss/dense_4_loss/Mean
Ą
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/MaximumMaximumAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum/y*)
_class
loc:@loss/dense_4_loss/Mean*
T0*#
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
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/ReshapeReshape:training/Adam/gradients/loss/dense_4_loss/mul_grad/ReshapeAtraining/Adam/gradients/loss/dense_4_loss/Mean_grad/DynamicStitch*
T0*
Tshape0*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
:

8training/Adam/gradients/loss/dense_4_loss/Mean_grad/TileTile;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Reshape<training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv*

Tmultiples0*)
_class
loc:@loss/dense_4_loss/Mean*
T0*
_output_shapes
:
˙
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_2ShapeYloss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean*
out_type0
ź
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3Shapeloss/dense_4_loss/Mean*
out_type0*
T0*
_output_shapes
:*)
_class
loc:@loss/dense_4_loss/Mean
Ž
9training/Adam/gradients/loss/dense_4_loss/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: *)
_class
loc:@loss/dense_4_loss/Mean
Ą
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/ProdProd;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_29training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const*
T0*)
_class
loc:@loss/dense_4_loss/Mean*
	keep_dims( *

Tidx0*
_output_shapes
: 
°
;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1Const*)
_class
loc:@loss/dense_4_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
Ľ
:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1Prod;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Shape_3;training/Adam/gradients/loss/dense_4_loss/Mean_grad/Const_1*
_output_shapes
: *
T0*

Tidx0*)
_class
loc:@loss/dense_4_loss/Mean*
	keep_dims( 
Ź
?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/yConst*)
_class
loc:@loss/dense_4_loss/Mean*
value	B :*
_output_shapes
: *
dtype0

=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1Maximum:training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod_1?training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1/y*
T0*
_output_shapes
: *)
_class
loc:@loss/dense_4_loss/Mean

>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1FloorDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Prod=training/Adam/gradients/loss/dense_4_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0*)
_class
loc:@loss/dense_4_loss/Mean
Ű
8training/Adam/gradients/loss/dense_4_loss/Mean_grad/CastCast>training/Adam/gradients/loss/dense_4_loss/Mean_grad/floordiv_1*)
_class
loc:@loss/dense_4_loss/Mean*
_output_shapes
: *

DstT0*

SrcT0

;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivRealDiv8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Tile8training/Adam/gradients/loss/dense_4_loss/Mean_grad/Cast*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*)
_class
loc:@loss/dense_4_loss/Mean*
T0
Ź
"training/Adam/gradients/zeros_like	ZerosLike[loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
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
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
: *
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0

training/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims;training/Adam/gradients/loss/dense_4_loss/Mean_grad/truedivtraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0
Ž
ztraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMultraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimstraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*l
_classb
`^loc:@loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ă
>training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/ShapeShapeloss/dense_4_loss/Log*.
_class$
" loc:@loss/dense_4_loss/Reshape_1*
T0*
out_type0*
_output_shapes
:
÷
@training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/ReshapeReshapeztraining/Adam/gradients/loss/dense_4_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul>training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Shape*
Tshape0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@loss/dense_4_loss/Reshape_1

=training/Adam/gradients/loss/dense_4_loss/Log_grad/Reciprocal
Reciprocalloss/dense_4_loss/clip_by_valueA^training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Reshape*(
_class
loc:@loss/dense_4_loss/Log*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

6training/Adam/gradients/loss/dense_4_loss/Log_grad/mulMul@training/Adam/gradients/loss/dense_4_loss/Reshape_1_grad/Reshape=training/Adam/gradients/loss/dense_4_loss/Log_grad/Reciprocal*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*(
_class
loc:@loss/dense_4_loss/Log
Ý
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ShapeShape'loss/dense_4_loss/clip_by_value/Minimum*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
_output_shapes
:*
out_type0*
T0
ť
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1Const*
dtype0*
valueB *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
_output_shapes
: 
î
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_2Shape6training/Adam/gradients/loss/dense_4_loss/Log_grad/mul*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
_output_shapes
:*
T0*
out_type0
Á
Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros/ConstConst*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
valueB
 *    *
dtype0*
_output_shapes
: 
Ŕ
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zerosFillDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_2Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value

Itraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqualGreaterEqual'loss/dense_4_loss/clip_by_value/Minimumloss/dense_4_loss/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
ć
Rtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ShapeDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ú
Ctraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SelectSelectItraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqual6training/Adam/gradients/loss/dense_4_loss/Log_grad/mulBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
ü
Etraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Select_1SelectItraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/GreaterEqualBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/zeros6training/Adam/gradients/loss/dense_4_loss/Log_grad/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
Ô
@training/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SumSumCtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SelectRtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgs*
T0*
	keep_dims( *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*

Tidx0*
_output_shapes
:
É
Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ReshapeReshape@training/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/SumBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape*
Tshape0*
T0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ú
Btraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Sum_1SumEtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Select_1Ttraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( *2
_class(
&$loc:@loss/dense_4_loss/clip_by_value
ž
Ftraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape_1ReshapeBtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Sum_1Dtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Shape_1*
_output_shapes
: *
Tshape0*2
_class(
&$loc:@loss/dense_4_loss/clip_by_value*
T0
Ő
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ShapeShapedense_4/Softmax*
T0*
_output_shapes
:*
out_type0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
Ë
Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*
_output_shapes
: *:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
valueB 

Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_2ShapeDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape*
out_type0*
_output_shapes
:*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
Ń
Ptraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
ŕ
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zerosFillLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_2Ptraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
ń
Ntraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualdense_4/Softmaxloss/dense_4_loss/sub*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ztraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ShapeLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
Ktraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SelectSelectNtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqualDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/ReshapeJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
T0
§
Mtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Select_1SelectNtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/LessEqualJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/zerosDtraining/Adam/gradients/loss/dense_4_loss/clip_by_value_grad/Reshape*
T0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ô
Htraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SumSumKtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SelectZtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*

Tidx0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
	keep_dims( *
T0*
_output_shapes
:
é
Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/ReshapeReshapeHtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/SumJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ú
Jtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Sum_1SumMtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Select_1\training/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum
Ţ
Ntraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeJtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Sum_1Ltraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Shape_1*:
_class0
.,loc:@loss/dense_4_loss/clip_by_value/Minimum*
_output_shapes
: *
T0*
Tshape0
ě
0training/Adam/gradients/dense_4/Softmax_grad/mulMulLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshapedense_4/Softmax*"
_class
loc:@dense_4/Softmax*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
°
Btraining/Adam/gradients/dense_4/Softmax_grad/Sum/reduction_indicesConst*"
_class
loc:@dense_4/Softmax*
_output_shapes
:*
valueB:*
dtype0

0training/Adam/gradients/dense_4/Softmax_grad/SumSum0training/Adam/gradients/dense_4/Softmax_grad/mulBtraining/Adam/gradients/dense_4/Softmax_grad/Sum/reduction_indices*

Tidx0*"
_class
loc:@dense_4/Softmax*
	keep_dims( *
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
:training/Adam/gradients/dense_4/Softmax_grad/Reshape/shapeConst*"
_class
loc:@dense_4/Softmax*
_output_shapes
:*
valueB"˙˙˙˙   *
dtype0

4training/Adam/gradients/dense_4/Softmax_grad/ReshapeReshape0training/Adam/gradients/dense_4/Softmax_grad/Sum:training/Adam/gradients/dense_4/Softmax_grad/Reshape/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0*"
_class
loc:@dense_4/Softmax

0training/Adam/gradients/dense_4/Softmax_grad/subSubLtraining/Adam/gradients/loss/dense_4_loss/clip_by_value/Minimum_grad/Reshape4training/Adam/gradients/dense_4/Softmax_grad/Reshape*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ň
2training/Adam/gradients/dense_4/Softmax_grad/mul_1Mul0training/Adam/gradients/dense_4/Softmax_grad/subdense_4/Softmax*
T0*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ű
8training/Adam/gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_4/Softmax_grad/mul_1*
T0*
_output_shapes
:*"
_class
loc:@dense_4/BiasAdd*
data_formatNHWC

2training/Adam/gradients/dense_4/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_4/Softmax_grad/mul_1dense_3/kernel/read*
T0*
transpose_b(*!
_class
loc:@dense_4/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 
ó
4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Relu2training/Adam/gradients/dense_4/Softmax_grad/mul_1*!
_class
loc:@dense_4/MatMul*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	
Ô
2training/Adam/gradients/dense_3/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_4/MatMul_grad/MatMuldense_3/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
_class
loc:@dense_3/Relu
Ü
8training/Adam/gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC*"
_class
loc:@dense_3/BiasAdd

2training/Adam/gradients/dense_3/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_3/Relu_grad/ReluGraddense_2/kernel/read*!
_class
loc:@dense_3/MatMul*
T0*
transpose_a( *
transpose_b(*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ô
4training/Adam/gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu2training/Adam/gradients/dense_3/Relu_grad/ReluGrad*
transpose_b( * 
_output_shapes
:
*
transpose_a(*!
_class
loc:@dense_3/MatMul*
T0
Ô
2training/Adam/gradients/dense_2/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_3/MatMul_grad/MatMuldense_2/Relu*
_class
loc:@dense_2/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ü
8training/Adam/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
data_formatNHWC*"
_class
loc:@dense_2/BiasAdd*
T0*
_output_shapes	
:

2training/Adam/gradients/dense_2/MatMul_grad/MatMulMatMul2training/Adam/gradients/dense_2/Relu_grad/ReluGraddense_1/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*!
_class
loc:@dense_2/MatMul*
T0*
transpose_a( 
ň
4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/Relu2training/Adam/gradients/dense_2/Relu_grad/ReluGrad*
transpose_b( * 
_output_shapes
:
*!
_class
loc:@dense_2/MatMul*
T0*
transpose_a(
Î
0training/Adam/gradients/dense/Relu_grad/ReluGradReluGrad2training/Adam/gradients/dense_2/MatMul_grad/MatMul
dense/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@dense/Relu
Ö
6training/Adam/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad0training/Adam/gradients/dense/Relu_grad/ReluGrad* 
_class
loc:@dense/BiasAdd*
T0*
data_formatNHWC*
_output_shapes	
:
ř
0training/Adam/gradients/dense/MatMul_grad/MatMulMatMul0training/Adam/gradients/dense/Relu_grad/ReluGraddense/kernel/read*
transpose_a( *
_class
loc:@dense/MatMul*
transpose_b(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
T0
î
2training/Adam/gradients/dense/MatMul_grad/MatMul_1MatMuldense_1_input0training/Adam/gradients/dense/Relu_grad/ReluGrad*
_output_shapes
:	1*
transpose_b( *
transpose_a(*
T0*
_class
loc:@dense/MatMul
_
training/Adam/AssignAdd/valueConst*
_output_shapes
: *
dtype0	*
value	B	 R
Ź
training/Adam/AssignAdd	AssignAddAdam/iterationstraining/Adam/AssignAdd/value*
T0	*"
_class
loc:@Adam/iterations*
_output_shapes
: *
use_locking( 
`
training/Adam/CastCastAdam/iterations/read*
_output_shapes
: *

SrcT0	*

DstT0
X
training/Adam/add/yConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
b
training/Adam/addAddtraining/Adam/Casttraining/Adam/add/y*
T0*
_output_shapes
: 
^
training/Adam/PowPowAdam/beta_2/readtraining/Adam/add*
T0*
_output_shapes
: 
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
training/Adam/ConstConst*
valueB
 *    *
_output_shapes
: *
dtype0
Z
training/Adam/Const_1Const*
_output_shapes
: *
valueB
 *  *
dtype0
y
#training/Adam/clip_by_value/MinimumMinimumtraining/Adam/subtraining/Adam/Const_1*
_output_shapes
: *
T0

training/Adam/clip_by_valueMaximum#training/Adam/clip_by_value/Minimumtraining/Adam/Const*
_output_shapes
: *
T0
X
training/Adam/SqrtSqrttraining/Adam/clip_by_value*
_output_shapes
: *
T0
`
training/Adam/Pow_1PowAdam/beta_1/readtraining/Adam/add*
_output_shapes
: *
T0
Z
training/Adam/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
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
training/Adam/mulMulAdam/lr/readtraining/Adam/truediv*
T0*
_output_shapes
: 
l
training/Adam/Const_2Const*
valueB	1*    *
dtype0*
_output_shapes
:	1

training/Adam/Variable
VariableV2*
_output_shapes
:	1*
dtype0*
shared_name *
shape:	1*
	container 
Ô
training/Adam/Variable/AssignAssigntraining/Adam/Variabletraining/Adam/Const_2*
validate_shape(*
use_locking(*
_output_shapes
:	1*
T0*)
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
VariableV2*
	container *
dtype0*
shared_name *
shape:*
_output_shapes	
:
Ö
training/Adam/Variable_1/AssignAssigntraining/Adam/Variable_1training/Adam/Const_3*
validate_shape(*
T0*+
_class!
loc:@training/Adam/Variable_1*
use_locking(*
_output_shapes	
:

training/Adam/Variable_1/readIdentitytraining/Adam/Variable_1*
T0*+
_class!
loc:@training/Adam/Variable_1*
_output_shapes	
:
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
dtype0*
shape:
* 
_output_shapes
:
*
	container *
shared_name 
Ű
training/Adam/Variable_2/AssignAssigntraining/Adam/Variable_2training/Adam/Const_4*
use_locking(*
T0*
validate_shape(*+
_class!
loc:@training/Adam/Variable_2* 
_output_shapes
:


training/Adam/Variable_2/readIdentitytraining/Adam/Variable_2*
T0* 
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_2
d
training/Adam/Const_5Const*
valueB*    *
_output_shapes	
:*
dtype0

training/Adam/Variable_3
VariableV2*
shape:*
	container *
shared_name *
dtype0*
_output_shapes	
:
Ö
training/Adam/Variable_3/AssignAssigntraining/Adam/Variable_3training/Adam/Const_5*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_3

training/Adam/Variable_3/readIdentitytraining/Adam/Variable_3*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_3*
T0
n
training/Adam/Const_6Const*
valueB
*    * 
_output_shapes
:
*
dtype0

training/Adam/Variable_4
VariableV2*
dtype0*
shared_name *
	container * 
_output_shapes
:
*
shape:

Ű
training/Adam/Variable_4/AssignAssigntraining/Adam/Variable_4training/Adam/Const_6*
validate_shape(*+
_class!
loc:@training/Adam/Variable_4*
use_locking(* 
_output_shapes
:
*
T0

training/Adam/Variable_4/readIdentitytraining/Adam/Variable_4*
T0* 
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_4
d
training/Adam/Const_7Const*
valueB*    *
_output_shapes	
:*
dtype0

training/Adam/Variable_5
VariableV2*
shared_name *
_output_shapes	
:*
dtype0*
shape:*
	container 
Ö
training/Adam/Variable_5/AssignAssigntraining/Adam/Variable_5training/Adam/Const_7*
_output_shapes	
:*
T0*
use_locking(*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(

training/Adam/Variable_5/readIdentitytraining/Adam/Variable_5*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_5
l
training/Adam/Const_8Const*
_output_shapes
:	*
dtype0*
valueB	*    

training/Adam/Variable_6
VariableV2*
dtype0*
shape:	*
shared_name *
_output_shapes
:	*
	container 
Ú
training/Adam/Variable_6/AssignAssigntraining/Adam/Variable_6training/Adam/Const_8*
use_locking(*+
_class!
loc:@training/Adam/Variable_6*
T0*
_output_shapes
:	*
validate_shape(

training/Adam/Variable_6/readIdentitytraining/Adam/Variable_6*
T0*
_output_shapes
:	*+
_class!
loc:@training/Adam/Variable_6
b
training/Adam/Const_9Const*
valueB*    *
dtype0*
_output_shapes
:

training/Adam/Variable_7
VariableV2*
_output_shapes
:*
shape:*
dtype0*
shared_name *
	container 
Ő
training/Adam/Variable_7/AssignAssigntraining/Adam/Variable_7training/Adam/Const_9*
_output_shapes
:*
use_locking(*+
_class!
loc:@training/Adam/Variable_7*
T0*
validate_shape(

training/Adam/Variable_7/readIdentitytraining/Adam/Variable_7*
_output_shapes
:*
T0*+
_class!
loc:@training/Adam/Variable_7
m
training/Adam/Const_10Const*
valueB	1*    *
_output_shapes
:	1*
dtype0

training/Adam/Variable_8
VariableV2*
shape:	1*
dtype0*
	container *
_output_shapes
:	1*
shared_name 
Ű
training/Adam/Variable_8/AssignAssigntraining/Adam/Variable_8training/Adam/Const_10*+
_class!
loc:@training/Adam/Variable_8*
_output_shapes
:	1*
use_locking(*
validate_shape(*
T0

training/Adam/Variable_8/readIdentitytraining/Adam/Variable_8*
_output_shapes
:	1*
T0*+
_class!
loc:@training/Adam/Variable_8
e
training/Adam/Const_11Const*
_output_shapes	
:*
dtype0*
valueB*    

training/Adam/Variable_9
VariableV2*
_output_shapes	
:*
shape:*
shared_name *
	container *
dtype0
×
training/Adam/Variable_9/AssignAssigntraining/Adam/Variable_9training/Adam/Const_11*
validate_shape(*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_9*
use_locking(

training/Adam/Variable_9/readIdentitytraining/Adam/Variable_9*
_output_shapes	
:*
T0*+
_class!
loc:@training/Adam/Variable_9
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
	container *
shared_name *
dtype0*
shape:
* 
_output_shapes
:

ß
 training/Adam/Variable_10/AssignAssigntraining/Adam/Variable_10training/Adam/Const_12*
use_locking(* 
_output_shapes
:
*,
_class"
 loc:@training/Adam/Variable_10*
T0*
validate_shape(

training/Adam/Variable_10/readIdentitytraining/Adam/Variable_10* 
_output_shapes
:
*
T0*,
_class"
 loc:@training/Adam/Variable_10
e
training/Adam/Const_13Const*
dtype0*
valueB*    *
_output_shapes	
:

training/Adam/Variable_11
VariableV2*
dtype0*
	container *
_output_shapes	
:*
shape:*
shared_name 
Ú
 training/Adam/Variable_11/AssignAssigntraining/Adam/Variable_11training/Adam/Const_13*
use_locking(*,
_class"
 loc:@training/Adam/Variable_11*
T0*
validate_shape(*
_output_shapes	
:
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
dtype0* 
_output_shapes
:


training/Adam/Variable_12
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
	container *
shared_name 
ß
 training/Adam/Variable_12/AssignAssigntraining/Adam/Variable_12training/Adam/Const_14*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_12* 
_output_shapes
:
*
T0

training/Adam/Variable_12/readIdentitytraining/Adam/Variable_12*
T0* 
_output_shapes
:
*,
_class"
 loc:@training/Adam/Variable_12
e
training/Adam/Const_15Const*
valueB*    *
dtype0*
_output_shapes	
:

training/Adam/Variable_13
VariableV2*
shape:*
	container *
shared_name *
dtype0*
_output_shapes	
:
Ú
 training/Adam/Variable_13/AssignAssigntraining/Adam/Variable_13training/Adam/Const_15*
T0*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes	
:*
use_locking(

training/Adam/Variable_13/readIdentitytraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
T0*
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
VariableV2*
_output_shapes
:	*
shape:	*
dtype0*
	container *
shared_name 
Ţ
 training/Adam/Variable_14/AssignAssigntraining/Adam/Variable_14training/Adam/Const_16*
validate_shape(*
use_locking(*,
_class"
 loc:@training/Adam/Variable_14*
_output_shapes
:	*
T0

training/Adam/Variable_14/readIdentitytraining/Adam/Variable_14*
T0*
_output_shapes
:	*,
_class"
 loc:@training/Adam/Variable_14
c
training/Adam/Const_17Const*
_output_shapes
:*
valueB*    *
dtype0

training/Adam/Variable_15
VariableV2*
shape:*
_output_shapes
:*
	container *
dtype0*
shared_name 
Ů
 training/Adam/Variable_15/AssignAssigntraining/Adam/Variable_15training/Adam/Const_17*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*,
_class"
 loc:@training/Adam/Variable_15

training/Adam/Variable_15/readIdentitytraining/Adam/Variable_15*
T0*
_output_shapes
:*,
_class"
 loc:@training/Adam/Variable_15
s
training/Adam/mul_1MulAdam/beta_1/readtraining/Adam/Variable/read*
_output_shapes
:	1*
T0
Z
training/Adam/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
d
training/Adam/sub_2Subtraining/Adam/sub_2/xAdam/beta_1/read*
_output_shapes
: *
T0
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
training/Adam/mul_3MulAdam/beta_2/readtraining/Adam/Variable_8/read*
T0*
_output_shapes
:	1
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
training/Adam/SquareSquare2training/Adam/gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	1*
T0
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
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_19Const*
_output_shapes
: *
dtype0*
valueB
 *  
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
training/Adam/add_3Addtraining/Adam/Sqrt_1training/Adam/add_3/y*
T0*
_output_shapes
:	1
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
loc:@training/Adam/Variable*
validate_shape(*
use_locking(*
_output_shapes
:	1*
T0
Ď
training/Adam/Assign_1Assigntraining/Adam/Variable_8training/Adam/add_2*
use_locking(*+
_class!
loc:@training/Adam/Variable_8*
validate_shape(*
T0*
_output_shapes
:	1
ˇ
training/Adam/Assign_2Assigndense/kerneltraining/Adam/sub_4*
_output_shapes
:	1*
use_locking(*
_class
loc:@dense/kernel*
validate_shape(*
T0
q
training/Adam/mul_6MulAdam/beta_1/readtraining/Adam/Variable_1/read*
_output_shapes	
:*
T0
Z
training/Adam/sub_5/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
d
training/Adam/sub_5Subtraining/Adam/sub_5/xAdam/beta_1/read*
_output_shapes
: *
T0

training/Adam/mul_7Multraining/Adam/sub_56training/Adam/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
j
training/Adam/add_4Addtraining/Adam/mul_6training/Adam/mul_7*
T0*
_output_shapes	
:
q
training/Adam/mul_8MulAdam/beta_2/readtraining/Adam/Variable_9/read*
T0*
_output_shapes	
:
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
training/Adam/mul_10Multraining/Adam/multraining/Adam/add_4*
T0*
_output_shapes	
:
[
training/Adam/Const_20Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_21Const*
dtype0*
valueB
 *  *
_output_shapes
: 

%training/Adam/clip_by_value_2/MinimumMinimumtraining/Adam/add_5training/Adam/Const_21*
T0*
_output_shapes	
:

training/Adam/clip_by_value_2Maximum%training/Adam/clip_by_value_2/Minimumtraining/Adam/Const_20*
T0*
_output_shapes	
:
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
training/Adam/add_6Addtraining/Adam/Sqrt_2training/Adam/add_6/y*
T0*
_output_shapes	
:
s
training/Adam/truediv_2RealDivtraining/Adam/mul_10training/Adam/add_6*
_output_shapes	
:*
T0
j
training/Adam/sub_7Subdense/bias/readtraining/Adam/truediv_2*
T0*
_output_shapes	
:
Ë
training/Adam/Assign_3Assigntraining/Adam/Variable_1training/Adam/add_4*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_1*
validate_shape(*
T0*
use_locking(
Ë
training/Adam/Assign_4Assigntraining/Adam/Variable_9training/Adam/add_5*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
Ż
training/Adam/Assign_5Assign
dense/biastraining/Adam/sub_7*
T0*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
_output_shapes	
:
w
training/Adam/mul_11MulAdam/beta_1/readtraining/Adam/Variable_2/read*
T0* 
_output_shapes
:

Z
training/Adam/sub_8/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
d
training/Adam/sub_8Subtraining/Adam/sub_8/xAdam/beta_1/read*
_output_shapes
: *
T0
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
training/Adam/mul_13MulAdam/beta_2/readtraining/Adam/Variable_10/read*
T0* 
_output_shapes
:

Z
training/Adam/sub_9/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
d
training/Adam/sub_9Subtraining/Adam/sub_9/xAdam/beta_2/read*
_output_shapes
: *
T0

training/Adam/Square_2Square4training/Adam/gradients/dense_2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

s
training/Adam/mul_14Multraining/Adam/sub_9training/Adam/Square_2*
T0* 
_output_shapes
:

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
training/Adam/Const_22Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 
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
training/Adam/add_9/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
r
training/Adam/add_9Addtraining/Adam/Sqrt_3training/Adam/add_9/y* 
_output_shapes
:
*
T0
x
training/Adam/truediv_3RealDivtraining/Adam/mul_15training/Adam/add_9*
T0* 
_output_shapes
:

t
training/Adam/sub_10Subdense_1/kernel/readtraining/Adam/truediv_3*
T0* 
_output_shapes
:

Đ
training/Adam/Assign_6Assigntraining/Adam/Variable_2training/Adam/add_7* 
_output_shapes
:
*+
_class!
loc:@training/Adam/Variable_2*
validate_shape(*
use_locking(*
T0
Ň
training/Adam/Assign_7Assigntraining/Adam/Variable_10training/Adam/add_8* 
_output_shapes
:
*
validate_shape(*
T0*
use_locking(*,
_class"
 loc:@training/Adam/Variable_10
˝
training/Adam/Assign_8Assigndense_1/kerneltraining/Adam/sub_10*
use_locking(* 
_output_shapes
:
*!
_class
loc:@dense_1/kernel*
validate_shape(*
T0
r
training/Adam/mul_16MulAdam/beta_1/readtraining/Adam/Variable_3/read*
T0*
_output_shapes	
:
[
training/Adam/sub_11/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
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
training/Adam/add_11Addtraining/Adam/mul_18training/Adam/mul_19*
T0*
_output_shapes	
:
j
training/Adam/mul_20Multraining/Adam/multraining/Adam/add_10*
_output_shapes	
:*
T0
[
training/Adam/Const_24Const*
dtype0*
_output_shapes
: *
valueB
 *    
[
training/Adam/Const_25Const*
dtype0*
valueB
 *  *
_output_shapes
: 
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
training/Adam/Sqrt_4Sqrttraining/Adam/clip_by_value_4*
_output_shapes	
:*
T0
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
use_locking(*
validate_shape(*
_output_shapes	
:*+
_class!
loc:@training/Adam/Variable_3*
T0
Ď
training/Adam/Assign_10Assigntraining/Adam/Variable_11training/Adam/add_11*,
_class"
 loc:@training/Adam/Variable_11*
T0*
_output_shapes	
:*
validate_shape(*
use_locking(
ľ
training/Adam/Assign_11Assigndense_1/biastraining/Adam/sub_13*
_output_shapes	
:*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
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
training/Adam/sub_14Subtraining/Adam/sub_14/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_22Multraining/Adam/sub_144training/Adam/gradients/dense_3/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
r
training/Adam/add_13Addtraining/Adam/mul_21training/Adam/mul_22* 
_output_shapes
:
*
T0
x
training/Adam/mul_23MulAdam/beta_2/readtraining/Adam/Variable_12/read* 
_output_shapes
:
*
T0
[
training/Adam/sub_15/xConst*
valueB
 *  ?*
dtype0*
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
training/Adam/mul_25Multraining/Adam/multraining/Adam/add_13* 
_output_shapes
:
*
T0
[
training/Adam/Const_26Const*
_output_shapes
: *
valueB
 *    *
dtype0
[
training/Adam/Const_27Const*
valueB
 *  *
_output_shapes
: *
dtype0

%training/Adam/clip_by_value_5/MinimumMinimumtraining/Adam/add_14training/Adam/Const_27* 
_output_shapes
:
*
T0

training/Adam/clip_by_value_5Maximum%training/Adam/clip_by_value_5/Minimumtraining/Adam/Const_26*
T0* 
_output_shapes
:

f
training/Adam/Sqrt_5Sqrttraining/Adam/clip_by_value_5* 
_output_shapes
:
*
T0
[
training/Adam/add_15/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
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
training/Adam/Assign_12Assigntraining/Adam/Variable_4training/Adam/add_13*+
_class!
loc:@training/Adam/Variable_4*
T0*
validate_shape(*
use_locking(* 
_output_shapes
:

Ô
training/Adam/Assign_13Assigntraining/Adam/Variable_12training/Adam/add_14*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(*,
_class"
 loc:@training/Adam/Variable_12
ž
training/Adam/Assign_14Assigndense_2/kerneltraining/Adam/sub_16*!
_class
loc:@dense_2/kernel* 
_output_shapes
:
*
use_locking(*
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
: *
valueB
 *  ?*
dtype0
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
training/Adam/sub_18/xConst*
dtype0*
_output_shapes
: *
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
training/Adam/Const_28Const*
dtype0*
valueB
 *    *
_output_shapes
: 
[
training/Adam/Const_29Const*
dtype0*
_output_shapes
: *
valueB
 *  

%training/Adam/clip_by_value_6/MinimumMinimumtraining/Adam/add_17training/Adam/Const_29*
T0*
_output_shapes	
:
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
training/Adam/add_18/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
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
training/Adam/sub_19Subdense_2/bias/readtraining/Adam/truediv_6*
T0*
_output_shapes	
:
Í
training/Adam/Assign_15Assigntraining/Adam/Variable_5training/Adam/add_16*
T0*
_output_shapes	
:*
use_locking(*+
_class!
loc:@training/Adam/Variable_5*
validate_shape(
Ď
training/Adam/Assign_16Assigntraining/Adam/Variable_13training/Adam/add_17*
_output_shapes	
:*,
_class"
 loc:@training/Adam/Variable_13*
validate_shape(*
T0*
use_locking(
ľ
training/Adam/Assign_17Assigndense_2/biastraining/Adam/sub_19*
validate_shape(*
use_locking(*
T0*
_class
loc:@dense_2/bias*
_output_shapes	
:
v
training/Adam/mul_31MulAdam/beta_1/readtraining/Adam/Variable_6/read*
T0*
_output_shapes
:	
[
training/Adam/sub_20/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
f
training/Adam/sub_20Subtraining/Adam/sub_20/xAdam/beta_1/read*
T0*
_output_shapes
: 

training/Adam/mul_32Multraining/Adam/sub_204training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
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
training/Adam/sub_21/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
training/Adam/sub_21Subtraining/Adam/sub_21/xAdam/beta_2/read*
T0*
_output_shapes
: 

training/Adam/Square_6Square4training/Adam/gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes
:	*
T0
s
training/Adam/mul_34Multraining/Adam/sub_21training/Adam/Square_6*
T0*
_output_shapes
:	
q
training/Adam/add_20Addtraining/Adam/mul_33training/Adam/mul_34*
T0*
_output_shapes
:	
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
training/Adam/Const_31Const*
dtype0*
valueB
 *  *
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
training/Adam/add_21/yConst*
valueB
 *wĚ+2*
_output_shapes
: *
dtype0
s
training/Adam/add_21Addtraining/Adam/Sqrt_7training/Adam/add_21/y*
_output_shapes
:	*
T0
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
:	*
validate_shape(*
T0*+
_class!
loc:@training/Adam/Variable_6*
use_locking(
Ó
training/Adam/Assign_19Assigntraining/Adam/Variable_14training/Adam/add_20*,
_class"
 loc:@training/Adam/Variable_14*
use_locking(*
_output_shapes
:	*
validate_shape(*
T0
˝
training/Adam/Assign_20Assigndense_3/kerneltraining/Adam/sub_22*
use_locking(*
validate_shape(*
_output_shapes
:	*
T0*!
_class
loc:@dense_3/kernel
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
training/Adam/sub_23Subtraining/Adam/sub_23/xAdam/beta_1/read*
_output_shapes
: *
T0
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
training/Adam/mul_38MulAdam/beta_2/readtraining/Adam/Variable_15/read*
_output_shapes
:*
T0
[
training/Adam/sub_24/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
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
training/Adam/mul_40Multraining/Adam/multraining/Adam/add_22*
T0*
_output_shapes
:
[
training/Adam/Const_32Const*
dtype0*
valueB
 *    *
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
%training/Adam/clip_by_value_8/MinimumMinimumtraining/Adam/add_23training/Adam/Const_33*
_output_shapes
:*
T0
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
training/Adam/add_24/yConst*
dtype0*
valueB
 *wĚ+2*
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
training/Adam/sub_25Subdense_3/bias/readtraining/Adam/truediv_8*
_output_shapes
:*
T0
Ě
training/Adam/Assign_21Assigntraining/Adam/Variable_7training/Adam/add_22*+
_class!
loc:@training/Adam/Variable_7*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
Î
training/Adam/Assign_22Assigntraining/Adam/Variable_15training/Adam/add_23*,
_class"
 loc:@training/Adam/Variable_15*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
´
training/Adam/Assign_23Assigndense_3/biastraining/Adam/sub_25*
use_locking(*
validate_shape(*
_class
loc:@dense_3/bias*
T0*
_output_shapes
:
ˇ
training/group_depsNoOp	^loss/mul^metrics/acc/Mean^training/Adam/AssignAdd^training/Adam/Assign^training/Adam/Assign_1^training/Adam/Assign_2^training/Adam/Assign_3^training/Adam/Assign_4^training/Adam/Assign_5^training/Adam/Assign_6^training/Adam/Assign_7^training/Adam/Assign_8^training/Adam/Assign_9^training/Adam/Assign_10^training/Adam/Assign_11^training/Adam/Assign_12^training/Adam/Assign_13^training/Adam/Assign_14^training/Adam/Assign_15^training/Adam/Assign_16^training/Adam/Assign_17^training/Adam/Assign_18^training/Adam/Assign_19^training/Adam/Assign_20^training/Adam/Assign_21^training/Adam/Assign_22^training/Adam/Assign_23
0

group_depsNoOp	^loss/mul^metrics/acc/Mean

IsVariableInitializedIsVariableInitializeddense/kernel*
_output_shapes
: *
_class
loc:@dense/kernel*
dtype0

IsVariableInitialized_1IsVariableInitialized
dense/bias*
_output_shapes
: *
_class
loc:@dense/bias*
dtype0

IsVariableInitialized_2IsVariableInitializeddense_1/kernel*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
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
IsVariableInitialized_5IsVariableInitializeddense_2/bias*
_output_shapes
: *
dtype0*
_class
loc:@dense_2/bias

IsVariableInitialized_6IsVariableInitializeddense_3/kernel*
dtype0*!
_class
loc:@dense_3/kernel*
_output_shapes
: 

IsVariableInitialized_7IsVariableInitializeddense_3/bias*
_output_shapes
: *
_class
loc:@dense_3/bias*
dtype0

IsVariableInitialized_8IsVariableInitializedAdam/iterations*
dtype0	*"
_class
loc:@Adam/iterations*
_output_shapes
: 
z
IsVariableInitialized_9IsVariableInitializedAdam/lr*
_output_shapes
: *
dtype0*
_class
loc:@Adam/lr

IsVariableInitialized_10IsVariableInitializedAdam/beta_1*
dtype0*
_output_shapes
: *
_class
loc:@Adam/beta_1
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
dtype0*
_class
loc:@Adam/decay*
_output_shapes
: 

IsVariableInitialized_13IsVariableInitializedtraining/Adam/Variable*)
_class
loc:@training/Adam/Variable*
_output_shapes
: *
dtype0

IsVariableInitialized_14IsVariableInitializedtraining/Adam/Variable_1*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_1*
dtype0

IsVariableInitialized_15IsVariableInitializedtraining/Adam/Variable_2*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_2*
dtype0

IsVariableInitialized_16IsVariableInitializedtraining/Adam/Variable_3*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_3*
dtype0

IsVariableInitialized_17IsVariableInitializedtraining/Adam/Variable_4*+
_class!
loc:@training/Adam/Variable_4*
dtype0*
_output_shapes
: 

IsVariableInitialized_18IsVariableInitializedtraining/Adam/Variable_5*
dtype0*+
_class!
loc:@training/Adam/Variable_5*
_output_shapes
: 

IsVariableInitialized_19IsVariableInitializedtraining/Adam/Variable_6*
_output_shapes
: *
dtype0*+
_class!
loc:@training/Adam/Variable_6

IsVariableInitialized_20IsVariableInitializedtraining/Adam/Variable_7*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_7

IsVariableInitialized_21IsVariableInitializedtraining/Adam/Variable_8*
dtype0*
_output_shapes
: *+
_class!
loc:@training/Adam/Variable_8

IsVariableInitialized_22IsVariableInitializedtraining/Adam/Variable_9*+
_class!
loc:@training/Adam/Variable_9*
_output_shapes
: *
dtype0

IsVariableInitialized_23IsVariableInitializedtraining/Adam/Variable_10*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_10

IsVariableInitialized_24IsVariableInitializedtraining/Adam/Variable_11*,
_class"
 loc:@training/Adam/Variable_11*
dtype0*
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
IsVariableInitialized_26IsVariableInitializedtraining/Adam/Variable_13*,
_class"
 loc:@training/Adam/Variable_13*
_output_shapes
: *
dtype0

IsVariableInitialized_27IsVariableInitializedtraining/Adam/Variable_14*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_14

IsVariableInitialized_28IsVariableInitializedtraining/Adam/Variable_15*
dtype0*
_output_shapes
: *,
_class"
 loc:@training/Adam/Variable_15
Ě
initNoOp^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^Adam/iterations/Assign^Adam/lr/Assign^Adam/beta_1/Assign^Adam/beta_2/Assign^Adam/decay/Assign^training/Adam/Variable/Assign ^training/Adam/Variable_1/Assign ^training/Adam/Variable_2/Assign ^training/Adam/Variable_3/Assign ^training/Adam/Variable_4/Assign ^training/Adam/Variable_5/Assign ^training/Adam/Variable_6/Assign ^training/Adam/Variable_7/Assign ^training/Adam/Variable_8/Assign ^training/Adam/Variable_9/Assign!^training/Adam/Variable_10/Assign!^training/Adam/Variable_11/Assign!^training/Adam/Variable_12/Assign!^training/Adam/Variable_13/Assign!^training/Adam/Variable_14/Assign!^training/Adam/Variable_15/Assign
p
dense_5_inputPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
shape:˙˙˙˙˙˙˙˙˙1
Ł
/dense_4/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_4/kernel*
valueB"1      *
dtype0*
_output_shapes
:

-dense_4/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_4/kernel*
valueB
 *<ž*
_output_shapes
: *
dtype0

-dense_4/kernel/Initializer/random_uniform/maxConst*!
_class
loc:@dense_4/kernel*
dtype0*
valueB
 *<>*
_output_shapes
: 
ě
7dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_4/kernel/Initializer/random_uniform/shape*

seed *!
_class
loc:@dense_4/kernel*
seed2 *
dtype0*
T0*
_output_shapes
:	1
Ö
-dense_4/kernel/Initializer/random_uniform/subSub-dense_4/kernel/Initializer/random_uniform/max-dense_4/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_4/kernel*
_output_shapes
: *
T0
é
-dense_4/kernel/Initializer/random_uniform/mulMul7dense_4/kernel/Initializer/random_uniform/RandomUniform-dense_4/kernel/Initializer/random_uniform/sub*
_output_shapes
:	1*!
_class
loc:@dense_4/kernel*
T0
Ű
)dense_4/kernel/Initializer/random_uniformAdd-dense_4/kernel/Initializer/random_uniform/mul-dense_4/kernel/Initializer/random_uniform/min*
_output_shapes
:	1*
T0*!
_class
loc:@dense_4/kernel
§
dense_4/kernel
VariableV2*
shape:	1*!
_class
loc:@dense_4/kernel*
	container *
_output_shapes
:	1*
shared_name *
dtype0
Đ
dense_4/kernel/AssignAssigndense_4/kernel)dense_4/kernel/Initializer/random_uniform*
use_locking(*
_output_shapes
:	1*
T0*!
_class
loc:@dense_4/kernel*
validate_shape(
|
dense_4/kernel/readIdentitydense_4/kernel*
_output_shapes
:	1*
T0*!
_class
loc:@dense_4/kernel

dense_4/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_4/bias*
dtype0*
_output_shapes	
:

dense_4/bias
VariableV2*
shape:*
_class
loc:@dense_4/bias*
	container *
_output_shapes	
:*
shared_name *
dtype0
ť
dense_4/bias/AssignAssigndense_4/biasdense_4/bias/Initializer/zeros*
_output_shapes	
:*
_class
loc:@dense_4/bias*
use_locking(*
validate_shape(*
T0
r
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes	
:*
T0

dense_5/MatMulMatMuldense_5_inputdense_4/kernel/read*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 

dense_5/BiasAddBiasAdddense_5/MatMuldense_4/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
dense_5/ReluReludense_5/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_5/kernel/Initializer/random_uniform/shapeConst*!
_class
loc:@dense_5/kernel*
_output_shapes
:*
dtype0*
valueB"      

-dense_5/kernel/Initializer/random_uniform/minConst*!
_class
loc:@dense_5/kernel*
valueB
 *   ž*
dtype0*
_output_shapes
: 

-dense_5/kernel/Initializer/random_uniform/maxConst*
valueB
 *   >*
_output_shapes
: *!
_class
loc:@dense_5/kernel*
dtype0
í
7dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_5/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_5/kernel*
T0*
dtype0*

seed *
seed2 * 
_output_shapes
:

Ö
-dense_5/kernel/Initializer/random_uniform/subSub-dense_5/kernel/Initializer/random_uniform/max-dense_5/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_5/kernel*
_output_shapes
: *
T0
ę
-dense_5/kernel/Initializer/random_uniform/mulMul7dense_5/kernel/Initializer/random_uniform/RandomUniform-dense_5/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_5/kernel* 
_output_shapes
:

Ü
)dense_5/kernel/Initializer/random_uniformAdd-dense_5/kernel/Initializer/random_uniform/mul-dense_5/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_5/kernel* 
_output_shapes
:

Š
dense_5/kernel
VariableV2* 
_output_shapes
:
*
shape:
*
shared_name *!
_class
loc:@dense_5/kernel*
	container *
dtype0
Ń
dense_5/kernel/AssignAssigndense_5/kernel)dense_5/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(* 
_output_shapes
:
*!
_class
loc:@dense_5/kernel*
T0
}
dense_5/kernel/readIdentitydense_5/kernel*!
_class
loc:@dense_5/kernel* 
_output_shapes
:
*
T0

dense_5/bias/Initializer/zerosConst*
_output_shapes	
:*
_class
loc:@dense_5/bias*
valueB*    *
dtype0

dense_5/bias
VariableV2*
_output_shapes	
:*
_class
loc:@dense_5/bias*
dtype0*
	container *
shared_name *
shape:
ť
dense_5/bias/AssignAssigndense_5/biasdense_5/bias/Initializer/zeros*
_class
loc:@dense_5/bias*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
r
dense_5/bias/readIdentitydense_5/bias*
_class
loc:@dense_5/bias*
T0*
_output_shapes	
:

dense_6/MatMulMatMuldense_5/Reludense_5/kernel/read*
transpose_b( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_a( 

dense_6/BiasAddBiasAdddense_6/MatMuldense_5/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
dense_6/ReluReludense_6/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_6/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *!
_class
loc:@dense_6/kernel

-dense_6/kernel/Initializer/random_uniform/minConst*
valueB
 *óľ˝*
_output_shapes
: *!
_class
loc:@dense_6/kernel*
dtype0

-dense_6/kernel/Initializer/random_uniform/maxConst*
valueB
 *óľ=*
_output_shapes
: *!
_class
loc:@dense_6/kernel*
dtype0
í
7dense_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_6/kernel/Initializer/random_uniform/shape*
seed2 *

seed * 
_output_shapes
:
*
dtype0*!
_class
loc:@dense_6/kernel*
T0
Ö
-dense_6/kernel/Initializer/random_uniform/subSub-dense_6/kernel/Initializer/random_uniform/max-dense_6/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_6/kernel*
_output_shapes
: *
T0
ę
-dense_6/kernel/Initializer/random_uniform/mulMul7dense_6/kernel/Initializer/random_uniform/RandomUniform-dense_6/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_6/kernel* 
_output_shapes
:

Ü
)dense_6/kernel/Initializer/random_uniformAdd-dense_6/kernel/Initializer/random_uniform/mul-dense_6/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*!
_class
loc:@dense_6/kernel
Š
dense_6/kernel
VariableV2*
	container *
dtype0*!
_class
loc:@dense_6/kernel*
shared_name *
shape:
* 
_output_shapes
:

Ń
dense_6/kernel/AssignAssigndense_6/kernel)dense_6/kernel/Initializer/random_uniform*
use_locking(* 
_output_shapes
:
*
validate_shape(*
T0*!
_class
loc:@dense_6/kernel
}
dense_6/kernel/readIdentitydense_6/kernel* 
_output_shapes
:
*!
_class
loc:@dense_6/kernel*
T0

dense_6/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_6/bias*
valueB*    *
_output_shapes	
:

dense_6/bias
VariableV2*
	container *
shared_name *
shape:*
_class
loc:@dense_6/bias*
_output_shapes	
:*
dtype0
ť
dense_6/bias/AssignAssigndense_6/biasdense_6/bias/Initializer/zeros*
use_locking(*
_output_shapes	
:*
_class
loc:@dense_6/bias*
T0*
validate_shape(
r
dense_6/bias/readIdentitydense_6/bias*
_class
loc:@dense_6/bias*
_output_shapes	
:*
T0

dense_7/MatMulMatMuldense_6/Reludense_6/kernel/read*
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 

dense_7/BiasAddBiasAdddense_7/MatMuldense_6/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
X
dense_7/ReluReludense_7/BiasAdd*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ł
/dense_7/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"   
   *
dtype0*!
_class
loc:@dense_7/kernel

-dense_7/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *Ű˝*!
_class
loc:@dense_7/kernel*
_output_shapes
: 

-dense_7/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ű=*!
_class
loc:@dense_7/kernel*
dtype0*
_output_shapes
: 
ě
7dense_7/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_7/kernel/Initializer/random_uniform/shape*!
_class
loc:@dense_7/kernel*
T0*

seed *
seed2 *
dtype0*
_output_shapes
:	

Ö
-dense_7/kernel/Initializer/random_uniform/subSub-dense_7/kernel/Initializer/random_uniform/max-dense_7/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_7/kernel*
T0*
_output_shapes
: 
é
-dense_7/kernel/Initializer/random_uniform/mulMul7dense_7/kernel/Initializer/random_uniform/RandomUniform-dense_7/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	
*!
_class
loc:@dense_7/kernel
Ű
)dense_7/kernel/Initializer/random_uniformAdd-dense_7/kernel/Initializer/random_uniform/mul-dense_7/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_7/kernel*
_output_shapes
:	

§
dense_7/kernel
VariableV2*!
_class
loc:@dense_7/kernel*
shape:	
*
shared_name *
dtype0*
	container *
_output_shapes
:	

Đ
dense_7/kernel/AssignAssigndense_7/kernel)dense_7/kernel/Initializer/random_uniform*
T0*
use_locking(*!
_class
loc:@dense_7/kernel*
validate_shape(*
_output_shapes
:	

|
dense_7/kernel/readIdentitydense_7/kernel*!
_class
loc:@dense_7/kernel*
_output_shapes
:	
*
T0

dense_7/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_7/bias*
_output_shapes
:
*
valueB
*    

dense_7/bias
VariableV2*
shape:
*
dtype0*
	container *
_class
loc:@dense_7/bias*
_output_shapes
:
*
shared_name 
ş
dense_7/bias/AssignAssigndense_7/biasdense_7/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
_class
loc:@dense_7/bias*
T0
q
dense_7/bias/readIdentitydense_7/bias*
T0*
_output_shapes
:
*
_class
loc:@dense_7/bias

dense_8/MatMulMatMuldense_7/Reludense_7/kernel/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*
transpose_a( *
transpose_b( 

dense_8/BiasAddBiasAdddense_8/MatMuldense_7/bias/read*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
data_formatNHWC
]
dense_8/SoftmaxSoftmaxdense_8/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

a
Adam_1/iterations/initial_valueConst*
_output_shapes
: *
value	B	 R *
dtype0	
u
Adam_1/iterations
VariableV2*
	container *
_output_shapes
: *
shared_name *
shape: *
dtype0	
Ć
Adam_1/iterations/AssignAssignAdam_1/iterationsAdam_1/iterations/initial_value*$
_class
loc:@Adam_1/iterations*
T0	*
_output_shapes
: *
use_locking(*
validate_shape(
|
Adam_1/iterations/readIdentityAdam_1/iterations*
T0	*$
_class
loc:@Adam_1/iterations*
_output_shapes
: 
\
Adam_1/lr/initial_valueConst*
_output_shapes
: *
valueB
 *ˇŃ8*
dtype0
m
	Adam_1/lr
VariableV2*
	container *
shape: *
shared_name *
dtype0*
_output_shapes
: 
Ś
Adam_1/lr/AssignAssign	Adam_1/lrAdam_1/lr/initial_value*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@Adam_1/lr
d
Adam_1/lr/readIdentity	Adam_1/lr*
_output_shapes
: *
T0*
_class
loc:@Adam_1/lr
`
Adam_1/beta_1/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
q
Adam_1/beta_1
VariableV2*
dtype0*
	container *
shape: *
shared_name *
_output_shapes
: 
ś
Adam_1/beta_1/AssignAssignAdam_1/beta_1Adam_1/beta_1/initial_value*
T0*
use_locking(*
validate_shape(*
_output_shapes
: * 
_class
loc:@Adam_1/beta_1
p
Adam_1/beta_1/readIdentityAdam_1/beta_1* 
_class
loc:@Adam_1/beta_1*
_output_shapes
: *
T0
`
Adam_1/beta_2/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *wž?
q
Adam_1/beta_2
VariableV2*
shared_name *
dtype0*
shape: *
	container *
_output_shapes
: 
ś
Adam_1/beta_2/AssignAssignAdam_1/beta_2Adam_1/beta_2/initial_value* 
_class
loc:@Adam_1/beta_2*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
p
Adam_1/beta_2/readIdentityAdam_1/beta_2* 
_class
loc:@Adam_1/beta_2*
T0*
_output_shapes
: 
_
Adam_1/decay/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
p
Adam_1/decay
VariableV2*
shape: *
dtype0*
	container *
_output_shapes
: *
shared_name 
˛
Adam_1/decay/AssignAssignAdam_1/decayAdam_1/decay/initial_value*
T0*
use_locking(*
_class
loc:@Adam_1/decay*
_output_shapes
: *
validate_shape(
m
Adam_1/decay/readIdentityAdam_1/decay*
_output_shapes
: *
T0*
_class
loc:@Adam_1/decay

dense_8_targetPlaceholder*
dtype0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
q
dense_8_sample_weightsPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
loss_1/dense_8_loss/ConstConst*
valueB
 *żÖ3*
dtype0*
_output_shapes
: 
^
loss_1/dense_8_loss/sub/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
u
loss_1/dense_8_loss/subSubloss_1/dense_8_loss/sub/xloss_1/dense_8_loss/Const*
_output_shapes
: *
T0

)loss_1/dense_8_loss/clip_by_value/MinimumMinimumdense_8/Softmaxloss_1/dense_8_loss/sub*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

¤
!loss_1/dense_8_loss/clip_by_valueMaximum)loss_1/dense_8_loss/clip_by_value/Minimumloss_1/dense_8_loss/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0
s
loss_1/dense_8_loss/LogLog!loss_1/dense_8_loss/clip_by_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

t
!loss_1/dense_8_loss/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
˙˙˙˙˙˙˙˙˙

loss_1/dense_8_loss/ReshapeReshapedense_8_target!loss_1/dense_8_loss/Reshape/shape*
Tshape0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
z
loss_1/dense_8_loss/CastCastloss_1/dense_8_loss/Reshape*

DstT0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0
t
#loss_1/dense_8_loss/Reshape_1/shapeConst*
valueB"˙˙˙˙
   *
_output_shapes
:*
dtype0
Ś
loss_1/dense_8_loss/Reshape_1Reshapeloss_1/dense_8_loss/Log#loss_1/dense_8_loss/Reshape_1/shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
Tshape0*
T0

=loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss_1/dense_8_loss/Cast*
T0	*
_output_shapes
:*
out_type0

[loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss_1/dense_8_loss/Reshape_1loss_1/dense_8_loss/Cast*
Tlabels0	*6
_output_shapes$
":˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*
T0
m
*loss_1/dense_8_loss/Mean/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
ô
loss_1/dense_8_loss/MeanMean[loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*loss_1/dense_8_loss/Mean/reduction_indices*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( *

Tidx0*
T0
~
loss_1/dense_8_loss/mulMulloss_1/dense_8_loss/Meandense_8_sample_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
loss_1/dense_8_loss/NotEqual/yConst*
_output_shapes
: *
valueB
 *    *
dtype0

loss_1/dense_8_loss/NotEqualNotEqualdense_8_sample_weightsloss_1/dense_8_loss/NotEqual/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
}
loss_1/dense_8_loss/Cast_1Castloss_1/dense_8_loss/NotEqual*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0

e
loss_1/dense_8_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

loss_1/dense_8_loss/Mean_1Meanloss_1/dense_8_loss/Cast_1loss_1/dense_8_loss/Const_1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 

loss_1/dense_8_loss/truedivRealDivloss_1/dense_8_loss/mulloss_1/dense_8_loss/Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
e
loss_1/dense_8_loss/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 

loss_1/dense_8_loss/Mean_2Meanloss_1/dense_8_loss/truedivloss_1/dense_8_loss/Const_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Q
loss_1/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
\

loss_1/mulMulloss_1/mul/xloss_1/dense_8_loss/Mean_2*
T0*
_output_shapes
: 
n
#metrics_1/acc/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙

metrics_1/acc/MaxMaxdense_8_target#metrics_1/acc/Max/reduction_indices*

Tidx0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims( 
i
metrics_1/acc/ArgMax/dimensionConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

metrics_1/acc/ArgMaxArgMaxdense_8/Softmaxmetrics_1/acc/ArgMax/dimension*
T0*

Tidx0*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
metrics_1/acc/CastCastmetrics_1/acc/ArgMax*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

SrcT0	
q
metrics_1/acc/EqualEqualmetrics_1/acc/Maxmetrics_1/acc/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
metrics_1/acc/Cast_1Castmetrics_1/acc/Equal*

SrcT0
*

DstT0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
metrics_1/acc/ConstConst*
valueB: *
_output_shapes
:*
dtype0

metrics_1/acc/MeanMeanmetrics_1/acc/Cast_1metrics_1/acc/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0

training_1/Adam/gradients/ShapeConst*
valueB *
_output_shapes
: *
_class
loc:@loss_1/mul*
dtype0

#training_1/Adam/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ?*
_class
loc:@loss_1/mul*
dtype0
Ź
training_1/Adam/gradients/FillFilltraining_1/Adam/gradients/Shape#training_1/Adam/gradients/grad_ys_0*
_output_shapes
: *
_class
loc:@loss_1/mul*
T0
°
-training_1/Adam/gradients/loss_1/mul_grad/MulMultraining_1/Adam/gradients/Fillloss_1/dense_8_loss/Mean_2*
_output_shapes
: *
_class
loc:@loss_1/mul*
T0
¤
/training_1/Adam/gradients/loss_1/mul_grad/Mul_1Multraining_1/Adam/gradients/Fillloss_1/mul/x*
_output_shapes
: *
T0*
_class
loc:@loss_1/mul
Ŕ
Gtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Reshape/shapeConst*
dtype0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
valueB:*
_output_shapes
:
¨
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ReshapeReshape/training_1/Adam/gradients/loss_1/mul_grad/Mul_1Gtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Reshape/shape*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
T0*
_output_shapes
:*
Tshape0
É
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ShapeShapeloss_1/dense_8_loss/truediv*
out_type0*
_output_shapes
:*
T0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
š
>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/TileTileAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Reshape?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape*

Tmultiples0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
Ë
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape_1Shapeloss_1/dense_8_loss/truediv*
_output_shapes
:*
out_type0*
T0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
ł
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape_2Const*
valueB *
dtype0*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
_output_shapes
: 
¸
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ConstConst*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
valueB: *
_output_shapes
:*
dtype0
ˇ
>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ProdProdAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape_1?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Const*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
ş
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Const_1Const*
valueB: *-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
dtype0*
_output_shapes
:
ť
@training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Prod_1ProdAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Shape_2Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Const_1*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
´
Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Maximum/yConst*
_output_shapes
: *-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
dtype0*
value	B :
Ł
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/MaximumMaximum@training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Prod_1Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Maximum/y*
_output_shapes
: *-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
T0
Ą
Btraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/floordivFloorDiv>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/ProdAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Maximum*
T0*
_output_shapes
: *-
_class#
!loc:@loss_1/dense_8_loss/Mean_2
é
>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/CastCastBtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/floordiv*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*
_output_shapes
: *

DstT0*

SrcT0
Š
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/truedivRealDiv>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Tile>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/Cast*-
_class#
!loc:@loss_1/dense_8_loss/Mean_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ç
@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/ShapeShapeloss_1/dense_8_loss/mul*
out_type0*
T0*
_output_shapes
:*.
_class$
" loc:@loss_1/dense_8_loss/truediv
ľ
Btraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *.
_class$
" loc:@loss_1/dense_8_loss/truediv
Ü
Ptraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/ShapeBtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*.
_class$
" loc:@loss_1/dense_8_loss/truediv

Btraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDivRealDivAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/truedivloss_1/dense_8_loss/Mean_1*.
_class$
" loc:@loss_1/dense_8_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ë
>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/SumSumBtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDivPtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/BroadcastGradientArgs*.
_class$
" loc:@loss_1/dense_8_loss/truediv*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
ť
Btraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/ReshapeReshape>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Sum@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*.
_class$
" loc:@loss_1/dense_8_loss/truediv*
T0
ź
>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/NegNegloss_1/dense_8_loss/mul*
T0*.
_class$
" loc:@loss_1/dense_8_loss/truediv*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

Dtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDiv_1RealDiv>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Negloss_1/dense_8_loss/Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*.
_class$
" loc:@loss_1/dense_8_loss/truediv

Dtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDiv_2RealDivDtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDiv_1loss_1/dense_8_loss/Mean_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*.
_class$
" loc:@loss_1/dense_8_loss/truediv
Ź
>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/mulMulAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_2_grad/truedivDtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/RealDiv_2*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*.
_class$
" loc:@loss_1/dense_8_loss/truediv
Ë
@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Sum_1Sum>training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/mulRtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *.
_class$
" loc:@loss_1/dense_8_loss/truediv*
T0*

Tidx0*
_output_shapes
:
´
Dtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Reshape_1Reshape@training_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Sum_1Btraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: *.
_class$
" loc:@loss_1/dense_8_loss/truediv
Ŕ
<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/ShapeShapeloss_1/dense_8_loss/Mean*
out_type0*
_output_shapes
:*
T0**
_class 
loc:@loss_1/dense_8_loss/mul
Ŕ
>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape_1Shapedense_8_sample_weights*
out_type0*
_output_shapes
:**
_class 
loc:@loss_1/dense_8_loss/mul*
T0
Ě
Ltraining_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙**
_class 
loc:@loss_1/dense_8_loss/mul*
T0
÷
:training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/mulMulBtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Reshapedense_8_sample_weights*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@loss_1/dense_8_loss/mul
ˇ
:training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/SumSum:training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/mulLtraining_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0**
_class 
loc:@loss_1/dense_8_loss/mul*

Tidx0*
_output_shapes
:
Ť
>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/ReshapeReshape:training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Sum<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙**
_class 
loc:@loss_1/dense_8_loss/mul
ű
<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/mul_1Mulloss_1/dense_8_loss/MeanBtraining_1/Adam/gradients/loss_1/dense_8_loss/truediv_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0**
_class 
loc:@loss_1/dense_8_loss/mul
˝
<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Sum_1Sum<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/mul_1Ntraining_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0**
_class 
loc:@loss_1/dense_8_loss/mul
ą
@training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Reshape_1Reshape<training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Sum_1>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/Shape_1**
_class 
loc:@loss_1/dense_8_loss/mul*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ShapeShape[loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*+
_class!
loc:@loss_1/dense_8_loss/Mean*
T0*
out_type0
Ť
<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean
ü
;training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/addAdd*loss_1/dense_8_loss/Mean/reduction_indices<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Size*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
: 

;training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/modFloorMod;training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/add<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Size*
_output_shapes
: *
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean
ś
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_1Const*
dtype0*
_output_shapes
:*+
_class!
loc:@loss_1/dense_8_loss/Mean*
valueB: 
˛
Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *+
_class!
loc:@loss_1/dense_8_loss/Mean
˛
Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*+
_class!
loc:@loss_1/dense_8_loss/Mean
ă
=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/rangeRangeCtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range/start<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/SizeCtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range/delta*

Tidx0*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
:
ą
Btraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Fill/valueConst*
value	B :*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
: *
dtype0

<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/FillFill?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_1Btraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Fill/value*
T0*
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean
ł
Etraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/DynamicStitchDynamicStitch=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/range;training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/mod=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Fill*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
N*+
_class!
loc:@loss_1/dense_8_loss/Mean
°
Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum/yConst*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
: *
value	B :*
dtype0
Ż
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/MaximumMaximumEtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/DynamicStitchAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum/y*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
§
@training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/floordivFloorDiv=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ż
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ReshapeReshape>training_1/Adam/gradients/loss_1/dense_8_loss/mul_grad/ReshapeEtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/DynamicStitch*
Tshape0*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
:
Š
<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/TileTile?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Reshape@training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/floordiv*

Tmultiples0*
_output_shapes
:*
T0*+
_class!
loc:@loss_1/dense_8_loss/Mean

?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_2Shape[loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:*+
_class!
loc:@loss_1/dense_8_loss/Mean
Ä
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_3Shapeloss_1/dense_8_loss/Mean*
T0*
out_type0*+
_class!
loc:@loss_1/dense_8_loss/Mean*
_output_shapes
:
´
=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: *+
_class!
loc:@loss_1/dense_8_loss/Mean
Ż
<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ProdProd?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_2=training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Const*
T0*
	keep_dims( *
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean*

Tidx0
ś
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:*+
_class!
loc:@loss_1/dense_8_loss/Mean
ł
>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Prod_1Prod?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Shape_3?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Const_1*
	keep_dims( *
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean*

Tidx0*
T0
˛
Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :*+
_class!
loc:@loss_1/dense_8_loss/Mean

Atraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum_1Maximum>training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Prod_1Ctraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum_1/y*
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean*
T0

Btraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/floordiv_1FloorDiv<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/ProdAtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Maximum_1*
T0*
_output_shapes
: *+
_class!
loc:@loss_1/dense_8_loss/Mean
ĺ
<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/CastCastBtraining_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/floordiv_1*+
_class!
loc:@loss_1/dense_8_loss/Mean*

SrcT0*
_output_shapes
: *

DstT0
Ą
?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/truedivRealDiv<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Tile<training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/Cast*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*+
_class!
loc:@loss_1/dense_8_loss/Mean
˛
$training_1/Adam/gradients/zeros_like	ZerosLike]loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
Ö
training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient]loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*´
message¨ĽCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
Ĺ
training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
_output_shapes
: *n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
dtype0

training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims?training_1/Adam/gradients/loss_1/dense_8_loss/Mean_grad/truedivtraining_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*

Tdim0*
T0
ź
~training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMultraining_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimstraining_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*n
_classd
b`loc:@loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits
Ë
Btraining_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/ShapeShapeloss_1/dense_8_loss/Log*
_output_shapes
:*
out_type0*0
_class&
$"loc:@loss_1/dense_8_loss/Reshape_1*
T0

Dtraining_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/ReshapeReshape~training_1/Adam/gradients/loss_1/dense_8_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulBtraining_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
Tshape0*
T0*0
_class&
$"loc:@loss_1/dense_8_loss/Reshape_1

Atraining_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/Reciprocal
Reciprocal!loss_1/dense_8_loss/clip_by_valueE^training_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/Reshape**
_class 
loc:@loss_1/dense_8_loss/Log*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

¨
:training_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/mulMulDtraining_1/Adam/gradients/loss_1/dense_8_loss/Reshape_1_grad/ReshapeAtraining_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/Reciprocal*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
**
_class 
loc:@loss_1/dense_8_loss/Log
ĺ
Ftraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/ShapeShape)loss_1/dense_8_loss/clip_by_value/Minimum*
T0*
out_type0*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
_output_shapes
:
Á
Htraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: *4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value
ř
Htraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_2Shape:training_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/mul*
T0*
out_type0*
_output_shapes
:*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value
Ç
Ltraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zeros/ConstConst*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
_output_shapes
: *
dtype0*
valueB
 *    
Î
Ftraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zerosFillHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_2Ltraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zeros/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value

Mtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/GreaterEqualGreaterEqual)loss_1/dense_8_loss/clip_by_value/Minimumloss_1/dense_8_loss/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
T0
ô
Vtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/ShapeHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_1*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0

Gtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/SelectSelectMtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/GreaterEqual:training_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/mulFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zeros*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value

Itraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Select_1SelectMtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/GreaterEqualFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/zeros:training_1/Adam/gradients/loss_1/dense_8_loss/Log_grad/mul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
T0
â
Dtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/SumSumGtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/SelectVtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( *4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value
×
Htraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/ReshapeReshapeDtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/SumFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
Tshape0*
T0
č
Ftraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Sum_1SumItraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Select_1Xtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
T0*
_output_shapes
:
Ě
Jtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Reshape_1ReshapeFtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Sum_1Htraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Shape_1*
T0*
Tshape0*4
_class*
(&loc:@loss_1/dense_8_loss/clip_by_value*
_output_shapes
: 
Ű
Ntraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/ShapeShapedense_8/Softmax*
out_type0*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
_output_shapes
:
Ń
Ptraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_1Const*
dtype0*
valueB *<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
_output_shapes
: 

Ptraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_2ShapeHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Reshape*
out_type0*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
_output_shapes
:
×
Ttraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zeros/ConstConst*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
_output_shapes
: *
dtype0*
valueB
 *    
î
Ntraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zerosFillPtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_2Ttraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zeros/Const*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

ů
Rtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/LessEqual	LessEqualdense_8/Softmaxloss_1/dense_8_loss/sub*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum

^training_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/ShapePtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_1*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
ˇ
Otraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/SelectSelectRtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/LessEqualHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/ReshapeNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zeros*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

š
Qtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Select_1SelectRtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/LessEqualNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/zerosHtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value_grad/Reshape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum

Ltraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/SumSumOtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Select^training_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
T0
÷
Ptraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/ReshapeReshapeLtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/SumNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
Tshape0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
T0

Ntraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Sum_1SumQtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Select_1`training_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*

Tidx0*
	keep_dims( *
T0
ě
Rtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Reshape_1ReshapeNtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Sum_1Ptraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Shape_1*
T0*<
_class2
0.loc:@loss_1/dense_8_loss/clip_by_value/Minimum*
_output_shapes
: *
Tshape0
ň
2training_1/Adam/gradients/dense_8/Softmax_grad/mulMulPtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Reshapedense_8/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*"
_class
loc:@dense_8/Softmax*
T0
˛
Dtraining_1/Adam/gradients/dense_8/Softmax_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*"
_class
loc:@dense_8/Softmax*
valueB:
˘
2training_1/Adam/gradients/dense_8/Softmax_grad/SumSum2training_1/Adam/gradients/dense_8/Softmax_grad/mulDtraining_1/Adam/gradients/dense_8/Softmax_grad/Sum/reduction_indices*"
_class
loc:@dense_8/Softmax*

Tidx0*
	keep_dims( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ą
<training_1/Adam/gradients/dense_8/Softmax_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"˙˙˙˙   *"
_class
loc:@dense_8/Softmax

6training_1/Adam/gradients/dense_8/Softmax_grad/ReshapeReshape2training_1/Adam/gradients/dense_8/Softmax_grad/Sum<training_1/Adam/gradients/dense_8/Softmax_grad/Reshape/shape*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*"
_class
loc:@dense_8/Softmax

2training_1/Adam/gradients/dense_8/Softmax_grad/subSubPtraining_1/Adam/gradients/loss_1/dense_8_loss/clip_by_value/Minimum_grad/Reshape6training_1/Adam/gradients/dense_8/Softmax_grad/Reshape*"
_class
loc:@dense_8/Softmax*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ö
4training_1/Adam/gradients/dense_8/Softmax_grad/mul_1Mul2training_1/Adam/gradients/dense_8/Softmax_grad/subdense_8/Softmax*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*"
_class
loc:@dense_8/Softmax*
T0
ß
:training_1/Adam/gradients/dense_8/BiasAdd_grad/BiasAddGradBiasAddGrad4training_1/Adam/gradients/dense_8/Softmax_grad/mul_1*
data_formatNHWC*
T0*"
_class
loc:@dense_8/BiasAdd*
_output_shapes
:


4training_1/Adam/gradients/dense_8/MatMul_grad/MatMulMatMul4training_1/Adam/gradients/dense_8/Softmax_grad/mul_1dense_7/kernel/read*!
_class
loc:@dense_8/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
transpose_a( *
T0
÷
6training_1/Adam/gradients/dense_8/MatMul_grad/MatMul_1MatMuldense_7/Relu4training_1/Adam/gradients/dense_8/Softmax_grad/mul_1*
T0*
transpose_a(*
transpose_b( *
_output_shapes
:	
*!
_class
loc:@dense_8/MatMul
Ř
4training_1/Adam/gradients/dense_7/Relu_grad/ReluGradReluGrad4training_1/Adam/gradients/dense_8/MatMul_grad/MatMuldense_7/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
_class
loc:@dense_7/Relu*
T0
ŕ
:training_1/Adam/gradients/dense_7/BiasAdd_grad/BiasAddGradBiasAddGrad4training_1/Adam/gradients/dense_7/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*"
_class
loc:@dense_7/BiasAdd*
data_formatNHWC

4training_1/Adam/gradients/dense_7/MatMul_grad/MatMulMatMul4training_1/Adam/gradients/dense_7/Relu_grad/ReluGraddense_6/kernel/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
transpose_a( *
T0*!
_class
loc:@dense_7/MatMul
ř
6training_1/Adam/gradients/dense_7/MatMul_grad/MatMul_1MatMuldense_6/Relu4training_1/Adam/gradients/dense_7/Relu_grad/ReluGrad* 
_output_shapes
:
*
transpose_b( *
transpose_a(*!
_class
loc:@dense_7/MatMul*
T0
Ř
4training_1/Adam/gradients/dense_6/Relu_grad/ReluGradReluGrad4training_1/Adam/gradients/dense_7/MatMul_grad/MatMuldense_6/Relu*
_class
loc:@dense_6/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
:training_1/Adam/gradients/dense_6/BiasAdd_grad/BiasAddGradBiasAddGrad4training_1/Adam/gradients/dense_6/Relu_grad/ReluGrad*
T0*
_output_shapes	
:*"
_class
loc:@dense_6/BiasAdd*
data_formatNHWC

4training_1/Adam/gradients/dense_6/MatMul_grad/MatMulMatMul4training_1/Adam/gradients/dense_6/Relu_grad/ReluGraddense_5/kernel/read*
transpose_b(*
transpose_a( *
T0*!
_class
loc:@dense_6/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ř
6training_1/Adam/gradients/dense_6/MatMul_grad/MatMul_1MatMuldense_5/Relu4training_1/Adam/gradients/dense_6/Relu_grad/ReluGrad* 
_output_shapes
:
*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_6/MatMul
Ř
4training_1/Adam/gradients/dense_5/Relu_grad/ReluGradReluGrad4training_1/Adam/gradients/dense_6/MatMul_grad/MatMuldense_5/Relu*
T0*
_class
loc:@dense_5/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ŕ
:training_1/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGradBiasAddGrad4training_1/Adam/gradients/dense_5/Relu_grad/ReluGrad*"
_class
loc:@dense_5/BiasAdd*
_output_shapes	
:*
T0*
data_formatNHWC

4training_1/Adam/gradients/dense_5/MatMul_grad/MatMulMatMul4training_1/Adam/gradients/dense_5/Relu_grad/ReluGraddense_4/kernel/read*!
_class
loc:@dense_5/MatMul*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙1*
transpose_b(
ř
6training_1/Adam/gradients/dense_5/MatMul_grad/MatMul_1MatMuldense_5_input4training_1/Adam/gradients/dense_5/Relu_grad/ReluGrad*!
_class
loc:@dense_5/MatMul*
transpose_a(*
transpose_b( *
_output_shapes
:	1*
T0
a
training_1/Adam/AssignAdd/valueConst*
_output_shapes
: *
value	B	 R*
dtype0	
´
training_1/Adam/AssignAdd	AssignAddAdam_1/iterationstraining_1/Adam/AssignAdd/value*$
_class
loc:@Adam_1/iterations*
T0	*
use_locking( *
_output_shapes
: 
d
training_1/Adam/CastCastAdam_1/iterations/read*
_output_shapes
: *

SrcT0	*

DstT0
Z
training_1/Adam/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
h
training_1/Adam/addAddtraining_1/Adam/Casttraining_1/Adam/add/y*
T0*
_output_shapes
: 
d
training_1/Adam/PowPowAdam_1/beta_2/readtraining_1/Adam/add*
T0*
_output_shapes
: 
Z
training_1/Adam/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
g
training_1/Adam/subSubtraining_1/Adam/sub/xtraining_1/Adam/Pow*
_output_shapes
: *
T0
Z
training_1/Adam/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
training_1/Adam/Const_1Const*
dtype0*
valueB
 *  *
_output_shapes
: 

%training_1/Adam/clip_by_value/MinimumMinimumtraining_1/Adam/subtraining_1/Adam/Const_1*
T0*
_output_shapes
: 

training_1/Adam/clip_by_valueMaximum%training_1/Adam/clip_by_value/Minimumtraining_1/Adam/Const*
T0*
_output_shapes
: 
\
training_1/Adam/SqrtSqrttraining_1/Adam/clip_by_value*
_output_shapes
: *
T0
f
training_1/Adam/Pow_1PowAdam_1/beta_1/readtraining_1/Adam/add*
T0*
_output_shapes
: 
\
training_1/Adam/sub_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
m
training_1/Adam/sub_1Subtraining_1/Adam/sub_1/xtraining_1/Adam/Pow_1*
T0*
_output_shapes
: 
p
training_1/Adam/truedivRealDivtraining_1/Adam/Sqrttraining_1/Adam/sub_1*
_output_shapes
: *
T0
d
training_1/Adam/mulMulAdam_1/lr/readtraining_1/Adam/truediv*
T0*
_output_shapes
: 
n
training_1/Adam/Const_2Const*
dtype0*
valueB	1*    *
_output_shapes
:	1

training_1/Adam/Variable
VariableV2*
_output_shapes
:	1*
	container *
dtype0*
shape:	1*
shared_name 
Ü
training_1/Adam/Variable/AssignAssigntraining_1/Adam/Variabletraining_1/Adam/Const_2*
use_locking(*
validate_shape(*
_output_shapes
:	1*+
_class!
loc:@training_1/Adam/Variable*
T0

training_1/Adam/Variable/readIdentitytraining_1/Adam/Variable*+
_class!
loc:@training_1/Adam/Variable*
T0*
_output_shapes
:	1
f
training_1/Adam/Const_3Const*
dtype0*
_output_shapes	
:*
valueB*    

training_1/Adam/Variable_1
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes	
:
Ţ
!training_1/Adam/Variable_1/AssignAssigntraining_1/Adam/Variable_1training_1/Adam/Const_3*
use_locking(*
validate_shape(*-
_class#
!loc:@training_1/Adam/Variable_1*
T0*
_output_shapes	
:

training_1/Adam/Variable_1/readIdentitytraining_1/Adam/Variable_1*-
_class#
!loc:@training_1/Adam/Variable_1*
_output_shapes	
:*
T0
p
training_1/Adam/Const_4Const*
dtype0* 
_output_shapes
:
*
valueB
*    

training_1/Adam/Variable_2
VariableV2*
shape:
* 
_output_shapes
:
*
shared_name *
dtype0*
	container 
ă
!training_1/Adam/Variable_2/AssignAssigntraining_1/Adam/Variable_2training_1/Adam/Const_4*
validate_shape(*
T0*-
_class#
!loc:@training_1/Adam/Variable_2*
use_locking(* 
_output_shapes
:

Ą
training_1/Adam/Variable_2/readIdentitytraining_1/Adam/Variable_2*-
_class#
!loc:@training_1/Adam/Variable_2*
T0* 
_output_shapes
:

f
training_1/Adam/Const_5Const*
valueB*    *
_output_shapes	
:*
dtype0

training_1/Adam/Variable_3
VariableV2*
	container *
dtype0*
shape:*
shared_name *
_output_shapes	
:
Ţ
!training_1/Adam/Variable_3/AssignAssigntraining_1/Adam/Variable_3training_1/Adam/Const_5*
_output_shapes	
:*-
_class#
!loc:@training_1/Adam/Variable_3*
use_locking(*
validate_shape(*
T0

training_1/Adam/Variable_3/readIdentitytraining_1/Adam/Variable_3*
_output_shapes	
:*
T0*-
_class#
!loc:@training_1/Adam/Variable_3
p
training_1/Adam/Const_6Const*
valueB
*    *
dtype0* 
_output_shapes
:


training_1/Adam/Variable_4
VariableV2*
	container *
shape:
*
shared_name *
dtype0* 
_output_shapes
:

ă
!training_1/Adam/Variable_4/AssignAssigntraining_1/Adam/Variable_4training_1/Adam/Const_6*
use_locking(*-
_class#
!loc:@training_1/Adam/Variable_4*
validate_shape(*
T0* 
_output_shapes
:

Ą
training_1/Adam/Variable_4/readIdentitytraining_1/Adam/Variable_4*-
_class#
!loc:@training_1/Adam/Variable_4* 
_output_shapes
:
*
T0
f
training_1/Adam/Const_7Const*
valueB*    *
dtype0*
_output_shapes	
:

training_1/Adam/Variable_5
VariableV2*
_output_shapes	
:*
shape:*
dtype0*
shared_name *
	container 
Ţ
!training_1/Adam/Variable_5/AssignAssigntraining_1/Adam/Variable_5training_1/Adam/Const_7*-
_class#
!loc:@training_1/Adam/Variable_5*
use_locking(*
_output_shapes	
:*
T0*
validate_shape(

training_1/Adam/Variable_5/readIdentitytraining_1/Adam/Variable_5*
T0*
_output_shapes	
:*-
_class#
!loc:@training_1/Adam/Variable_5
n
training_1/Adam/Const_8Const*
dtype0*
_output_shapes
:	
*
valueB	
*    

training_1/Adam/Variable_6
VariableV2*
_output_shapes
:	
*
shape:	
*
shared_name *
	container *
dtype0
â
!training_1/Adam/Variable_6/AssignAssigntraining_1/Adam/Variable_6training_1/Adam/Const_8*
T0*
use_locking(*-
_class#
!loc:@training_1/Adam/Variable_6*
_output_shapes
:	
*
validate_shape(
 
training_1/Adam/Variable_6/readIdentitytraining_1/Adam/Variable_6*-
_class#
!loc:@training_1/Adam/Variable_6*
T0*
_output_shapes
:	

d
training_1/Adam/Const_9Const*
dtype0*
_output_shapes
:
*
valueB
*    

training_1/Adam/Variable_7
VariableV2*
_output_shapes
:
*
dtype0*
	container *
shared_name *
shape:

Ý
!training_1/Adam/Variable_7/AssignAssigntraining_1/Adam/Variable_7training_1/Adam/Const_9*
_output_shapes
:
*
validate_shape(*
use_locking(*
T0*-
_class#
!loc:@training_1/Adam/Variable_7

training_1/Adam/Variable_7/readIdentitytraining_1/Adam/Variable_7*-
_class#
!loc:@training_1/Adam/Variable_7*
_output_shapes
:
*
T0
o
training_1/Adam/Const_10Const*
valueB	1*    *
dtype0*
_output_shapes
:	1

training_1/Adam/Variable_8
VariableV2*
_output_shapes
:	1*
shared_name *
shape:	1*
	container *
dtype0
ă
!training_1/Adam/Variable_8/AssignAssigntraining_1/Adam/Variable_8training_1/Adam/Const_10*
validate_shape(*
T0*
_output_shapes
:	1*
use_locking(*-
_class#
!loc:@training_1/Adam/Variable_8
 
training_1/Adam/Variable_8/readIdentitytraining_1/Adam/Variable_8*-
_class#
!loc:@training_1/Adam/Variable_8*
_output_shapes
:	1*
T0
g
training_1/Adam/Const_11Const*
dtype0*
valueB*    *
_output_shapes	
:

training_1/Adam/Variable_9
VariableV2*
dtype0*
shared_name *
	container *
shape:*
_output_shapes	
:
ß
!training_1/Adam/Variable_9/AssignAssigntraining_1/Adam/Variable_9training_1/Adam/Const_11*-
_class#
!loc:@training_1/Adam/Variable_9*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:

training_1/Adam/Variable_9/readIdentitytraining_1/Adam/Variable_9*
_output_shapes	
:*
T0*-
_class#
!loc:@training_1/Adam/Variable_9
q
training_1/Adam/Const_12Const*
dtype0*
valueB
*    * 
_output_shapes
:


training_1/Adam/Variable_10
VariableV2*
shared_name *
shape:
*
	container *
dtype0* 
_output_shapes
:

ç
"training_1/Adam/Variable_10/AssignAssigntraining_1/Adam/Variable_10training_1/Adam/Const_12*
validate_shape(* 
_output_shapes
:
*.
_class$
" loc:@training_1/Adam/Variable_10*
use_locking(*
T0
¤
 training_1/Adam/Variable_10/readIdentitytraining_1/Adam/Variable_10*.
_class$
" loc:@training_1/Adam/Variable_10* 
_output_shapes
:
*
T0
g
training_1/Adam/Const_13Const*
valueB*    *
_output_shapes	
:*
dtype0

training_1/Adam/Variable_11
VariableV2*
shape:*
shared_name *
	container *
_output_shapes	
:*
dtype0
â
"training_1/Adam/Variable_11/AssignAssigntraining_1/Adam/Variable_11training_1/Adam/Const_13*.
_class$
" loc:@training_1/Adam/Variable_11*
T0*
validate_shape(*
_output_shapes	
:*
use_locking(

 training_1/Adam/Variable_11/readIdentitytraining_1/Adam/Variable_11*.
_class$
" loc:@training_1/Adam/Variable_11*
_output_shapes	
:*
T0
q
training_1/Adam/Const_14Const*
dtype0* 
_output_shapes
:
*
valueB
*    

training_1/Adam/Variable_12
VariableV2*
shape:
*
	container *
dtype0*
shared_name * 
_output_shapes
:

ç
"training_1/Adam/Variable_12/AssignAssigntraining_1/Adam/Variable_12training_1/Adam/Const_14*.
_class$
" loc:@training_1/Adam/Variable_12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(
¤
 training_1/Adam/Variable_12/readIdentitytraining_1/Adam/Variable_12* 
_output_shapes
:
*.
_class$
" loc:@training_1/Adam/Variable_12*
T0
g
training_1/Adam/Const_15Const*
valueB*    *
dtype0*
_output_shapes	
:

training_1/Adam/Variable_13
VariableV2*
shape:*
	container *
_output_shapes	
:*
shared_name *
dtype0
â
"training_1/Adam/Variable_13/AssignAssigntraining_1/Adam/Variable_13training_1/Adam/Const_15*
validate_shape(*
_output_shapes	
:*.
_class$
" loc:@training_1/Adam/Variable_13*
T0*
use_locking(

 training_1/Adam/Variable_13/readIdentitytraining_1/Adam/Variable_13*
_output_shapes	
:*.
_class$
" loc:@training_1/Adam/Variable_13*
T0
o
training_1/Adam/Const_16Const*
valueB	
*    *
dtype0*
_output_shapes
:	


training_1/Adam/Variable_14
VariableV2*
dtype0*
shared_name *
_output_shapes
:	
*
	container *
shape:	

ć
"training_1/Adam/Variable_14/AssignAssigntraining_1/Adam/Variable_14training_1/Adam/Const_16*.
_class$
" loc:@training_1/Adam/Variable_14*
_output_shapes
:	
*
use_locking(*
validate_shape(*
T0
Ł
 training_1/Adam/Variable_14/readIdentitytraining_1/Adam/Variable_14*
T0*
_output_shapes
:	
*.
_class$
" loc:@training_1/Adam/Variable_14
e
training_1/Adam/Const_17Const*
_output_shapes
:
*
dtype0*
valueB
*    

training_1/Adam/Variable_15
VariableV2*
dtype0*
shape:
*
	container *
shared_name *
_output_shapes
:

á
"training_1/Adam/Variable_15/AssignAssigntraining_1/Adam/Variable_15training_1/Adam/Const_17*
T0*
_output_shapes
:
*
validate_shape(*.
_class$
" loc:@training_1/Adam/Variable_15*
use_locking(

 training_1/Adam/Variable_15/readIdentitytraining_1/Adam/Variable_15*
_output_shapes
:
*.
_class$
" loc:@training_1/Adam/Variable_15*
T0
y
training_1/Adam/mul_1MulAdam_1/beta_1/readtraining_1/Adam/Variable/read*
T0*
_output_shapes
:	1
\
training_1/Adam/sub_2/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
j
training_1/Adam/sub_2Subtraining_1/Adam/sub_2/xAdam_1/beta_1/read*
T0*
_output_shapes
: 

training_1/Adam/mul_2Multraining_1/Adam/sub_26training_1/Adam/gradients/dense_5/MatMul_grad/MatMul_1*
_output_shapes
:	1*
T0
t
training_1/Adam/add_1Addtraining_1/Adam/mul_1training_1/Adam/mul_2*
T0*
_output_shapes
:	1
{
training_1/Adam/mul_3MulAdam_1/beta_2/readtraining_1/Adam/Variable_8/read*
_output_shapes
:	1*
T0
\
training_1/Adam/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
j
training_1/Adam/sub_3Subtraining_1/Adam/sub_3/xAdam_1/beta_2/read*
T0*
_output_shapes
: 

training_1/Adam/SquareSquare6training_1/Adam/gradients/dense_5/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	1
u
training_1/Adam/mul_4Multraining_1/Adam/sub_3training_1/Adam/Square*
_output_shapes
:	1*
T0
t
training_1/Adam/add_2Addtraining_1/Adam/mul_3training_1/Adam/mul_4*
_output_shapes
:	1*
T0
r
training_1/Adam/mul_5Multraining_1/Adam/multraining_1/Adam/add_1*
_output_shapes
:	1*
T0
]
training_1/Adam/Const_18Const*
_output_shapes
: *
dtype0*
valueB
 *    
]
training_1/Adam/Const_19Const*
_output_shapes
: *
valueB
 *  *
dtype0

'training_1/Adam/clip_by_value_1/MinimumMinimumtraining_1/Adam/add_2training_1/Adam/Const_19*
T0*
_output_shapes
:	1

training_1/Adam/clip_by_value_1Maximum'training_1/Adam/clip_by_value_1/Minimumtraining_1/Adam/Const_18*
_output_shapes
:	1*
T0
i
training_1/Adam/Sqrt_1Sqrttraining_1/Adam/clip_by_value_1*
T0*
_output_shapes
:	1
\
training_1/Adam/add_3/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
w
training_1/Adam/add_3Addtraining_1/Adam/Sqrt_1training_1/Adam/add_3/y*
T0*
_output_shapes
:	1
|
training_1/Adam/truediv_1RealDivtraining_1/Adam/mul_5training_1/Adam/add_3*
T0*
_output_shapes
:	1
v
training_1/Adam/sub_4Subdense_4/kernel/readtraining_1/Adam/truediv_1*
_output_shapes
:	1*
T0
Ń
training_1/Adam/AssignAssigntraining_1/Adam/Variabletraining_1/Adam/add_1*
use_locking(*
_output_shapes
:	1*+
_class!
loc:@training_1/Adam/Variable*
T0*
validate_shape(
×
training_1/Adam/Assign_1Assigntraining_1/Adam/Variable_8training_1/Adam/add_2*
validate_shape(*
T0*-
_class#
!loc:@training_1/Adam/Variable_8*
use_locking(*
_output_shapes
:	1
ż
training_1/Adam/Assign_2Assigndense_4/kerneltraining_1/Adam/sub_4*
_output_shapes
:	1*
use_locking(*
T0*
validate_shape(*!
_class
loc:@dense_4/kernel
w
training_1/Adam/mul_6MulAdam_1/beta_1/readtraining_1/Adam/Variable_1/read*
T0*
_output_shapes	
:
\
training_1/Adam/sub_5/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
j
training_1/Adam/sub_5Subtraining_1/Adam/sub_5/xAdam_1/beta_1/read*
T0*
_output_shapes
: 

training_1/Adam/mul_7Multraining_1/Adam/sub_5:training_1/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
p
training_1/Adam/add_4Addtraining_1/Adam/mul_6training_1/Adam/mul_7*
_output_shapes	
:*
T0
w
training_1/Adam/mul_8MulAdam_1/beta_2/readtraining_1/Adam/Variable_9/read*
_output_shapes	
:*
T0
\
training_1/Adam/sub_6/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
j
training_1/Adam/sub_6Subtraining_1/Adam/sub_6/xAdam_1/beta_2/read*
_output_shapes
: *
T0

training_1/Adam/Square_1Square:training_1/Adam/gradients/dense_5/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
s
training_1/Adam/mul_9Multraining_1/Adam/sub_6training_1/Adam/Square_1*
T0*
_output_shapes	
:
p
training_1/Adam/add_5Addtraining_1/Adam/mul_8training_1/Adam/mul_9*
T0*
_output_shapes	
:
o
training_1/Adam/mul_10Multraining_1/Adam/multraining_1/Adam/add_4*
T0*
_output_shapes	
:
]
training_1/Adam/Const_20Const*
dtype0*
_output_shapes
: *
valueB
 *    
]
training_1/Adam/Const_21Const*
dtype0*
valueB
 *  *
_output_shapes
: 

'training_1/Adam/clip_by_value_2/MinimumMinimumtraining_1/Adam/add_5training_1/Adam/Const_21*
T0*
_output_shapes	
:

training_1/Adam/clip_by_value_2Maximum'training_1/Adam/clip_by_value_2/Minimumtraining_1/Adam/Const_20*
T0*
_output_shapes	
:
e
training_1/Adam/Sqrt_2Sqrttraining_1/Adam/clip_by_value_2*
_output_shapes	
:*
T0
\
training_1/Adam/add_6/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
s
training_1/Adam/add_6Addtraining_1/Adam/Sqrt_2training_1/Adam/add_6/y*
T0*
_output_shapes	
:
y
training_1/Adam/truediv_2RealDivtraining_1/Adam/mul_10training_1/Adam/add_6*
T0*
_output_shapes	
:
p
training_1/Adam/sub_7Subdense_4/bias/readtraining_1/Adam/truediv_2*
T0*
_output_shapes	
:
Ó
training_1/Adam/Assign_3Assigntraining_1/Adam/Variable_1training_1/Adam/add_4*-
_class#
!loc:@training_1/Adam/Variable_1*
_output_shapes	
:*
use_locking(*
T0*
validate_shape(
Ó
training_1/Adam/Assign_4Assigntraining_1/Adam/Variable_9training_1/Adam/add_5*
_output_shapes	
:*
validate_shape(*-
_class#
!loc:@training_1/Adam/Variable_9*
T0*
use_locking(
ˇ
training_1/Adam/Assign_5Assigndense_4/biastraining_1/Adam/sub_7*
_class
loc:@dense_4/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:
}
training_1/Adam/mul_11MulAdam_1/beta_1/readtraining_1/Adam/Variable_2/read* 
_output_shapes
:
*
T0
\
training_1/Adam/sub_8/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
j
training_1/Adam/sub_8Subtraining_1/Adam/sub_8/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_12Multraining_1/Adam/sub_86training_1/Adam/gradients/dense_6/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
w
training_1/Adam/add_7Addtraining_1/Adam/mul_11training_1/Adam/mul_12*
T0* 
_output_shapes
:

~
training_1/Adam/mul_13MulAdam_1/beta_2/read training_1/Adam/Variable_10/read* 
_output_shapes
:
*
T0
\
training_1/Adam/sub_9/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
j
training_1/Adam/sub_9Subtraining_1/Adam/sub_9/xAdam_1/beta_2/read*
T0*
_output_shapes
: 

training_1/Adam/Square_2Square6training_1/Adam/gradients/dense_6/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:

y
training_1/Adam/mul_14Multraining_1/Adam/sub_9training_1/Adam/Square_2* 
_output_shapes
:
*
T0
w
training_1/Adam/add_8Addtraining_1/Adam/mul_13training_1/Adam/mul_14*
T0* 
_output_shapes
:

t
training_1/Adam/mul_15Multraining_1/Adam/multraining_1/Adam/add_7* 
_output_shapes
:
*
T0
]
training_1/Adam/Const_22Const*
valueB
 *    *
_output_shapes
: *
dtype0
]
training_1/Adam/Const_23Const*
valueB
 *  *
dtype0*
_output_shapes
: 

'training_1/Adam/clip_by_value_3/MinimumMinimumtraining_1/Adam/add_8training_1/Adam/Const_23* 
_output_shapes
:
*
T0

training_1/Adam/clip_by_value_3Maximum'training_1/Adam/clip_by_value_3/Minimumtraining_1/Adam/Const_22*
T0* 
_output_shapes
:

j
training_1/Adam/Sqrt_3Sqrttraining_1/Adam/clip_by_value_3* 
_output_shapes
:
*
T0
\
training_1/Adam/add_9/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
x
training_1/Adam/add_9Addtraining_1/Adam/Sqrt_3training_1/Adam/add_9/y*
T0* 
_output_shapes
:

~
training_1/Adam/truediv_3RealDivtraining_1/Adam/mul_15training_1/Adam/add_9* 
_output_shapes
:
*
T0
x
training_1/Adam/sub_10Subdense_5/kernel/readtraining_1/Adam/truediv_3*
T0* 
_output_shapes
:

Ř
training_1/Adam/Assign_6Assigntraining_1/Adam/Variable_2training_1/Adam/add_7*
validate_shape(*
T0* 
_output_shapes
:
*-
_class#
!loc:@training_1/Adam/Variable_2*
use_locking(
Ú
training_1/Adam/Assign_7Assigntraining_1/Adam/Variable_10training_1/Adam/add_8*
validate_shape(*
use_locking(*.
_class$
" loc:@training_1/Adam/Variable_10*
T0* 
_output_shapes
:

Á
training_1/Adam/Assign_8Assigndense_5/kerneltraining_1/Adam/sub_10*
use_locking(*
T0* 
_output_shapes
:
*
validate_shape(*!
_class
loc:@dense_5/kernel
x
training_1/Adam/mul_16MulAdam_1/beta_1/readtraining_1/Adam/Variable_3/read*
T0*
_output_shapes	
:
]
training_1/Adam/sub_11/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
l
training_1/Adam/sub_11Subtraining_1/Adam/sub_11/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_17Multraining_1/Adam/sub_11:training_1/Adam/gradients/dense_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
s
training_1/Adam/add_10Addtraining_1/Adam/mul_16training_1/Adam/mul_17*
_output_shapes	
:*
T0
y
training_1/Adam/mul_18MulAdam_1/beta_2/read training_1/Adam/Variable_11/read*
_output_shapes	
:*
T0
]
training_1/Adam/sub_12/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
l
training_1/Adam/sub_12Subtraining_1/Adam/sub_12/xAdam_1/beta_2/read*
_output_shapes
: *
T0

training_1/Adam/Square_3Square:training_1/Adam/gradients/dense_6/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
u
training_1/Adam/mul_19Multraining_1/Adam/sub_12training_1/Adam/Square_3*
_output_shapes	
:*
T0
s
training_1/Adam/add_11Addtraining_1/Adam/mul_18training_1/Adam/mul_19*
_output_shapes	
:*
T0
p
training_1/Adam/mul_20Multraining_1/Adam/multraining_1/Adam/add_10*
_output_shapes	
:*
T0
]
training_1/Adam/Const_24Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training_1/Adam/Const_25Const*
dtype0*
_output_shapes
: *
valueB
 *  

'training_1/Adam/clip_by_value_4/MinimumMinimumtraining_1/Adam/add_11training_1/Adam/Const_25*
T0*
_output_shapes	
:

training_1/Adam/clip_by_value_4Maximum'training_1/Adam/clip_by_value_4/Minimumtraining_1/Adam/Const_24*
T0*
_output_shapes	
:
e
training_1/Adam/Sqrt_4Sqrttraining_1/Adam/clip_by_value_4*
_output_shapes	
:*
T0
]
training_1/Adam/add_12/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
u
training_1/Adam/add_12Addtraining_1/Adam/Sqrt_4training_1/Adam/add_12/y*
_output_shapes	
:*
T0
z
training_1/Adam/truediv_4RealDivtraining_1/Adam/mul_20training_1/Adam/add_12*
T0*
_output_shapes	
:
q
training_1/Adam/sub_13Subdense_5/bias/readtraining_1/Adam/truediv_4*
T0*
_output_shapes	
:
Ô
training_1/Adam/Assign_9Assigntraining_1/Adam/Variable_3training_1/Adam/add_10*
T0*
_output_shapes	
:*-
_class#
!loc:@training_1/Adam/Variable_3*
validate_shape(*
use_locking(
×
training_1/Adam/Assign_10Assigntraining_1/Adam/Variable_11training_1/Adam/add_11*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:*.
_class$
" loc:@training_1/Adam/Variable_11
š
training_1/Adam/Assign_11Assigndense_5/biastraining_1/Adam/sub_13*
use_locking(*
T0*
validate_shape(*
_class
loc:@dense_5/bias*
_output_shapes	
:
}
training_1/Adam/mul_21MulAdam_1/beta_1/readtraining_1/Adam/Variable_4/read* 
_output_shapes
:
*
T0
]
training_1/Adam/sub_14/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
l
training_1/Adam/sub_14Subtraining_1/Adam/sub_14/xAdam_1/beta_1/read*
T0*
_output_shapes
: 

training_1/Adam/mul_22Multraining_1/Adam/sub_146training_1/Adam/gradients/dense_7/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
x
training_1/Adam/add_13Addtraining_1/Adam/mul_21training_1/Adam/mul_22* 
_output_shapes
:
*
T0
~
training_1/Adam/mul_23MulAdam_1/beta_2/read training_1/Adam/Variable_12/read* 
_output_shapes
:
*
T0
]
training_1/Adam/sub_15/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
l
training_1/Adam/sub_15Subtraining_1/Adam/sub_15/xAdam_1/beta_2/read*
_output_shapes
: *
T0

training_1/Adam/Square_4Square6training_1/Adam/gradients/dense_7/MatMul_grad/MatMul_1* 
_output_shapes
:
*
T0
z
training_1/Adam/mul_24Multraining_1/Adam/sub_15training_1/Adam/Square_4*
T0* 
_output_shapes
:

x
training_1/Adam/add_14Addtraining_1/Adam/mul_23training_1/Adam/mul_24*
T0* 
_output_shapes
:

u
training_1/Adam/mul_25Multraining_1/Adam/multraining_1/Adam/add_13* 
_output_shapes
:
*
T0
]
training_1/Adam/Const_26Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training_1/Adam/Const_27Const*
_output_shapes
: *
dtype0*
valueB
 *  

'training_1/Adam/clip_by_value_5/MinimumMinimumtraining_1/Adam/add_14training_1/Adam/Const_27*
T0* 
_output_shapes
:


training_1/Adam/clip_by_value_5Maximum'training_1/Adam/clip_by_value_5/Minimumtraining_1/Adam/Const_26* 
_output_shapes
:
*
T0
j
training_1/Adam/Sqrt_5Sqrttraining_1/Adam/clip_by_value_5* 
_output_shapes
:
*
T0
]
training_1/Adam/add_15/yConst*
valueB
 *wĚ+2*
dtype0*
_output_shapes
: 
z
training_1/Adam/add_15Addtraining_1/Adam/Sqrt_5training_1/Adam/add_15/y*
T0* 
_output_shapes
:


training_1/Adam/truediv_5RealDivtraining_1/Adam/mul_25training_1/Adam/add_15* 
_output_shapes
:
*
T0
x
training_1/Adam/sub_16Subdense_6/kernel/readtraining_1/Adam/truediv_5*
T0* 
_output_shapes
:

Ú
training_1/Adam/Assign_12Assigntraining_1/Adam/Variable_4training_1/Adam/add_13*
T0*-
_class#
!loc:@training_1/Adam/Variable_4* 
_output_shapes
:
*
validate_shape(*
use_locking(
Ü
training_1/Adam/Assign_13Assigntraining_1/Adam/Variable_12training_1/Adam/add_14*.
_class$
" loc:@training_1/Adam/Variable_12*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:

Â
training_1/Adam/Assign_14Assigndense_6/kerneltraining_1/Adam/sub_16* 
_output_shapes
:
*
T0*
use_locking(*!
_class
loc:@dense_6/kernel*
validate_shape(
x
training_1/Adam/mul_26MulAdam_1/beta_1/readtraining_1/Adam/Variable_5/read*
_output_shapes	
:*
T0
]
training_1/Adam/sub_17/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
training_1/Adam/sub_17Subtraining_1/Adam/sub_17/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_27Multraining_1/Adam/sub_17:training_1/Adam/gradients/dense_7/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes	
:
s
training_1/Adam/add_16Addtraining_1/Adam/mul_26training_1/Adam/mul_27*
_output_shapes	
:*
T0
y
training_1/Adam/mul_28MulAdam_1/beta_2/read training_1/Adam/Variable_13/read*
T0*
_output_shapes	
:
]
training_1/Adam/sub_18/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
l
training_1/Adam/sub_18Subtraining_1/Adam/sub_18/xAdam_1/beta_2/read*
T0*
_output_shapes
: 

training_1/Adam/Square_5Square:training_1/Adam/gradients/dense_7/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
u
training_1/Adam/mul_29Multraining_1/Adam/sub_18training_1/Adam/Square_5*
T0*
_output_shapes	
:
s
training_1/Adam/add_17Addtraining_1/Adam/mul_28training_1/Adam/mul_29*
T0*
_output_shapes	
:
p
training_1/Adam/mul_30Multraining_1/Adam/multraining_1/Adam/add_16*
_output_shapes	
:*
T0
]
training_1/Adam/Const_28Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training_1/Adam/Const_29Const*
valueB
 *  *
dtype0*
_output_shapes
: 

'training_1/Adam/clip_by_value_6/MinimumMinimumtraining_1/Adam/add_17training_1/Adam/Const_29*
T0*
_output_shapes	
:

training_1/Adam/clip_by_value_6Maximum'training_1/Adam/clip_by_value_6/Minimumtraining_1/Adam/Const_28*
T0*
_output_shapes	
:
e
training_1/Adam/Sqrt_6Sqrttraining_1/Adam/clip_by_value_6*
_output_shapes	
:*
T0
]
training_1/Adam/add_18/yConst*
_output_shapes
: *
dtype0*
valueB
 *wĚ+2
u
training_1/Adam/add_18Addtraining_1/Adam/Sqrt_6training_1/Adam/add_18/y*
T0*
_output_shapes	
:
z
training_1/Adam/truediv_6RealDivtraining_1/Adam/mul_30training_1/Adam/add_18*
_output_shapes	
:*
T0
q
training_1/Adam/sub_19Subdense_6/bias/readtraining_1/Adam/truediv_6*
T0*
_output_shapes	
:
Ő
training_1/Adam/Assign_15Assigntraining_1/Adam/Variable_5training_1/Adam/add_16*
validate_shape(*-
_class#
!loc:@training_1/Adam/Variable_5*
T0*
use_locking(*
_output_shapes	
:
×
training_1/Adam/Assign_16Assigntraining_1/Adam/Variable_13training_1/Adam/add_17*.
_class$
" loc:@training_1/Adam/Variable_13*
T0*
use_locking(*
_output_shapes	
:*
validate_shape(
š
training_1/Adam/Assign_17Assigndense_6/biastraining_1/Adam/sub_19*
validate_shape(*
use_locking(*
_class
loc:@dense_6/bias*
_output_shapes	
:*
T0
|
training_1/Adam/mul_31MulAdam_1/beta_1/readtraining_1/Adam/Variable_6/read*
T0*
_output_shapes
:	

]
training_1/Adam/sub_20/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
l
training_1/Adam/sub_20Subtraining_1/Adam/sub_20/xAdam_1/beta_1/read*
T0*
_output_shapes
: 

training_1/Adam/mul_32Multraining_1/Adam/sub_206training_1/Adam/gradients/dense_8/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	

w
training_1/Adam/add_19Addtraining_1/Adam/mul_31training_1/Adam/mul_32*
_output_shapes
:	
*
T0
}
training_1/Adam/mul_33MulAdam_1/beta_2/read training_1/Adam/Variable_14/read*
T0*
_output_shapes
:	

]
training_1/Adam/sub_21/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
l
training_1/Adam/sub_21Subtraining_1/Adam/sub_21/xAdam_1/beta_2/read*
T0*
_output_shapes
: 

training_1/Adam/Square_6Square6training_1/Adam/gradients/dense_8/MatMul_grad/MatMul_1*
_output_shapes
:	
*
T0
y
training_1/Adam/mul_34Multraining_1/Adam/sub_21training_1/Adam/Square_6*
_output_shapes
:	
*
T0
w
training_1/Adam/add_20Addtraining_1/Adam/mul_33training_1/Adam/mul_34*
_output_shapes
:	
*
T0
t
training_1/Adam/mul_35Multraining_1/Adam/multraining_1/Adam/add_19*
T0*
_output_shapes
:	

]
training_1/Adam/Const_30Const*
_output_shapes
: *
dtype0*
valueB
 *    
]
training_1/Adam/Const_31Const*
_output_shapes
: *
valueB
 *  *
dtype0

'training_1/Adam/clip_by_value_7/MinimumMinimumtraining_1/Adam/add_20training_1/Adam/Const_31*
_output_shapes
:	
*
T0

training_1/Adam/clip_by_value_7Maximum'training_1/Adam/clip_by_value_7/Minimumtraining_1/Adam/Const_30*
T0*
_output_shapes
:	

i
training_1/Adam/Sqrt_7Sqrttraining_1/Adam/clip_by_value_7*
T0*
_output_shapes
:	

]
training_1/Adam/add_21/yConst*
_output_shapes
: *
valueB
 *wĚ+2*
dtype0
y
training_1/Adam/add_21Addtraining_1/Adam/Sqrt_7training_1/Adam/add_21/y*
T0*
_output_shapes
:	

~
training_1/Adam/truediv_7RealDivtraining_1/Adam/mul_35training_1/Adam/add_21*
_output_shapes
:	
*
T0
w
training_1/Adam/sub_22Subdense_7/kernel/readtraining_1/Adam/truediv_7*
_output_shapes
:	
*
T0
Ů
training_1/Adam/Assign_18Assigntraining_1/Adam/Variable_6training_1/Adam/add_19*
use_locking(*
validate_shape(*
_output_shapes
:	
*
T0*-
_class#
!loc:@training_1/Adam/Variable_6
Ű
training_1/Adam/Assign_19Assigntraining_1/Adam/Variable_14training_1/Adam/add_20*
_output_shapes
:	
*
T0*.
_class$
" loc:@training_1/Adam/Variable_14*
use_locking(*
validate_shape(
Á
training_1/Adam/Assign_20Assigndense_7/kerneltraining_1/Adam/sub_22*
T0*
_output_shapes
:	
*
validate_shape(*!
_class
loc:@dense_7/kernel*
use_locking(
w
training_1/Adam/mul_36MulAdam_1/beta_1/readtraining_1/Adam/Variable_7/read*
_output_shapes
:
*
T0
]
training_1/Adam/sub_23/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
l
training_1/Adam/sub_23Subtraining_1/Adam/sub_23/xAdam_1/beta_1/read*
_output_shapes
: *
T0

training_1/Adam/mul_37Multraining_1/Adam/sub_23:training_1/Adam/gradients/dense_8/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
r
training_1/Adam/add_22Addtraining_1/Adam/mul_36training_1/Adam/mul_37*
_output_shapes
:
*
T0
x
training_1/Adam/mul_38MulAdam_1/beta_2/read training_1/Adam/Variable_15/read*
_output_shapes
:
*
T0
]
training_1/Adam/sub_24/xConst*
valueB
 *  ?*
_output_shapes
: *
dtype0
l
training_1/Adam/sub_24Subtraining_1/Adam/sub_24/xAdam_1/beta_2/read*
_output_shapes
: *
T0

training_1/Adam/Square_7Square:training_1/Adam/gradients/dense_8/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
*
T0
t
training_1/Adam/mul_39Multraining_1/Adam/sub_24training_1/Adam/Square_7*
T0*
_output_shapes
:

r
training_1/Adam/add_23Addtraining_1/Adam/mul_38training_1/Adam/mul_39*
_output_shapes
:
*
T0
o
training_1/Adam/mul_40Multraining_1/Adam/multraining_1/Adam/add_22*
_output_shapes
:
*
T0
]
training_1/Adam/Const_32Const*
valueB
 *    *
dtype0*
_output_shapes
: 
]
training_1/Adam/Const_33Const*
_output_shapes
: *
dtype0*
valueB
 *  

'training_1/Adam/clip_by_value_8/MinimumMinimumtraining_1/Adam/add_23training_1/Adam/Const_33*
T0*
_output_shapes
:


training_1/Adam/clip_by_value_8Maximum'training_1/Adam/clip_by_value_8/Minimumtraining_1/Adam/Const_32*
_output_shapes
:
*
T0
d
training_1/Adam/Sqrt_8Sqrttraining_1/Adam/clip_by_value_8*
T0*
_output_shapes
:

]
training_1/Adam/add_24/yConst*
dtype0*
_output_shapes
: *
valueB
 *wĚ+2
t
training_1/Adam/add_24Addtraining_1/Adam/Sqrt_8training_1/Adam/add_24/y*
T0*
_output_shapes
:

y
training_1/Adam/truediv_8RealDivtraining_1/Adam/mul_40training_1/Adam/add_24*
_output_shapes
:
*
T0
p
training_1/Adam/sub_25Subdense_7/bias/readtraining_1/Adam/truediv_8*
_output_shapes
:
*
T0
Ô
training_1/Adam/Assign_21Assigntraining_1/Adam/Variable_7training_1/Adam/add_22*
validate_shape(*
use_locking(*-
_class#
!loc:@training_1/Adam/Variable_7*
T0*
_output_shapes
:

Ö
training_1/Adam/Assign_22Assigntraining_1/Adam/Variable_15training_1/Adam/add_23*
validate_shape(*
T0*
_output_shapes
:
*.
_class$
" loc:@training_1/Adam/Variable_15*
use_locking(
¸
training_1/Adam/Assign_23Assigndense_7/biastraining_1/Adam/sub_25*
use_locking(*
T0*
_output_shapes
:
*
validate_shape(*
_class
loc:@dense_7/bias
ď
training_1/group_depsNoOp^loss_1/mul^metrics_1/acc/Mean^training_1/Adam/AssignAdd^training_1/Adam/Assign^training_1/Adam/Assign_1^training_1/Adam/Assign_2^training_1/Adam/Assign_3^training_1/Adam/Assign_4^training_1/Adam/Assign_5^training_1/Adam/Assign_6^training_1/Adam/Assign_7^training_1/Adam/Assign_8^training_1/Adam/Assign_9^training_1/Adam/Assign_10^training_1/Adam/Assign_11^training_1/Adam/Assign_12^training_1/Adam/Assign_13^training_1/Adam/Assign_14^training_1/Adam/Assign_15^training_1/Adam/Assign_16^training_1/Adam/Assign_17^training_1/Adam/Assign_18^training_1/Adam/Assign_19^training_1/Adam/Assign_20^training_1/Adam/Assign_21^training_1/Adam/Assign_22^training_1/Adam/Assign_23
6
group_deps_1NoOp^loss_1/mul^metrics_1/acc/Mean

IsVariableInitialized_29IsVariableInitializeddense_4/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_4/kernel

IsVariableInitialized_30IsVariableInitializeddense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes
: *
dtype0

IsVariableInitialized_31IsVariableInitializeddense_5/kernel*
_output_shapes
: *!
_class
loc:@dense_5/kernel*
dtype0

IsVariableInitialized_32IsVariableInitializeddense_5/bias*
_output_shapes
: *
dtype0*
_class
loc:@dense_5/bias

IsVariableInitialized_33IsVariableInitializeddense_6/kernel*!
_class
loc:@dense_6/kernel*
_output_shapes
: *
dtype0

IsVariableInitialized_34IsVariableInitializeddense_6/bias*
_output_shapes
: *
dtype0*
_class
loc:@dense_6/bias

IsVariableInitialized_35IsVariableInitializeddense_7/kernel*
dtype0*!
_class
loc:@dense_7/kernel*
_output_shapes
: 

IsVariableInitialized_36IsVariableInitializeddense_7/bias*
_output_shapes
: *
_class
loc:@dense_7/bias*
dtype0

IsVariableInitialized_37IsVariableInitializedAdam_1/iterations*
_output_shapes
: *$
_class
loc:@Adam_1/iterations*
dtype0	

IsVariableInitialized_38IsVariableInitialized	Adam_1/lr*
_output_shapes
: *
_class
loc:@Adam_1/lr*
dtype0

IsVariableInitialized_39IsVariableInitializedAdam_1/beta_1*
dtype0*
_output_shapes
: * 
_class
loc:@Adam_1/beta_1

IsVariableInitialized_40IsVariableInitializedAdam_1/beta_2*
dtype0* 
_class
loc:@Adam_1/beta_2*
_output_shapes
: 

IsVariableInitialized_41IsVariableInitializedAdam_1/decay*
dtype0*
_class
loc:@Adam_1/decay*
_output_shapes
: 

IsVariableInitialized_42IsVariableInitializedtraining_1/Adam/Variable*
_output_shapes
: *+
_class!
loc:@training_1/Adam/Variable*
dtype0
Ą
IsVariableInitialized_43IsVariableInitializedtraining_1/Adam/Variable_1*
dtype0*
_output_shapes
: *-
_class#
!loc:@training_1/Adam/Variable_1
Ą
IsVariableInitialized_44IsVariableInitializedtraining_1/Adam/Variable_2*-
_class#
!loc:@training_1/Adam/Variable_2*
dtype0*
_output_shapes
: 
Ą
IsVariableInitialized_45IsVariableInitializedtraining_1/Adam/Variable_3*
_output_shapes
: *
dtype0*-
_class#
!loc:@training_1/Adam/Variable_3
Ą
IsVariableInitialized_46IsVariableInitializedtraining_1/Adam/Variable_4*
_output_shapes
: *
dtype0*-
_class#
!loc:@training_1/Adam/Variable_4
Ą
IsVariableInitialized_47IsVariableInitializedtraining_1/Adam/Variable_5*
dtype0*
_output_shapes
: *-
_class#
!loc:@training_1/Adam/Variable_5
Ą
IsVariableInitialized_48IsVariableInitializedtraining_1/Adam/Variable_6*-
_class#
!loc:@training_1/Adam/Variable_6*
dtype0*
_output_shapes
: 
Ą
IsVariableInitialized_49IsVariableInitializedtraining_1/Adam/Variable_7*
dtype0*-
_class#
!loc:@training_1/Adam/Variable_7*
_output_shapes
: 
Ą
IsVariableInitialized_50IsVariableInitializedtraining_1/Adam/Variable_8*
_output_shapes
: *-
_class#
!loc:@training_1/Adam/Variable_8*
dtype0
Ą
IsVariableInitialized_51IsVariableInitializedtraining_1/Adam/Variable_9*
_output_shapes
: *-
_class#
!loc:@training_1/Adam/Variable_9*
dtype0
Ł
IsVariableInitialized_52IsVariableInitializedtraining_1/Adam/Variable_10*
_output_shapes
: *
dtype0*.
_class$
" loc:@training_1/Adam/Variable_10
Ł
IsVariableInitialized_53IsVariableInitializedtraining_1/Adam/Variable_11*
dtype0*.
_class$
" loc:@training_1/Adam/Variable_11*
_output_shapes
: 
Ł
IsVariableInitialized_54IsVariableInitializedtraining_1/Adam/Variable_12*.
_class$
" loc:@training_1/Adam/Variable_12*
dtype0*
_output_shapes
: 
Ł
IsVariableInitialized_55IsVariableInitializedtraining_1/Adam/Variable_13*.
_class$
" loc:@training_1/Adam/Variable_13*
_output_shapes
: *
dtype0
Ł
IsVariableInitialized_56IsVariableInitializedtraining_1/Adam/Variable_14*.
_class$
" loc:@training_1/Adam/Variable_14*
_output_shapes
: *
dtype0
Ł
IsVariableInitialized_57IsVariableInitializedtraining_1/Adam/Variable_15*
_output_shapes
: *
dtype0*.
_class$
" loc:@training_1/Adam/Variable_15
ü
init_1NoOp^dense_4/kernel/Assign^dense_4/bias/Assign^dense_5/kernel/Assign^dense_5/bias/Assign^dense_6/kernel/Assign^dense_6/bias/Assign^dense_7/kernel/Assign^dense_7/bias/Assign^Adam_1/iterations/Assign^Adam_1/lr/Assign^Adam_1/beta_1/Assign^Adam_1/beta_2/Assign^Adam_1/decay/Assign ^training_1/Adam/Variable/Assign"^training_1/Adam/Variable_1/Assign"^training_1/Adam/Variable_2/Assign"^training_1/Adam/Variable_3/Assign"^training_1/Adam/Variable_4/Assign"^training_1/Adam/Variable_5/Assign"^training_1/Adam/Variable_6/Assign"^training_1/Adam/Variable_7/Assign"^training_1/Adam/Variable_8/Assign"^training_1/Adam/Variable_9/Assign#^training_1/Adam/Variable_10/Assign#^training_1/Adam/Variable_11/Assign#^training_1/Adam/Variable_12/Assign#^training_1/Adam/Variable_13/Assign#^training_1/Adam/Variable_14/Assign#^training_1/Adam/Variable_15/Assign""ż3
trainable_variables§3¤3
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
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/Const_17:0
m
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02+dense_4/kernel/Initializer/random_uniform:0
\
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02 dense_4/bias/Initializer/zeros:0
m
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:02+dense_5/kernel/Initializer/random_uniform:0
\
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:02 dense_5/bias/Initializer/zeros:0
m
dense_6/kernel:0dense_6/kernel/Assigndense_6/kernel/read:02+dense_6/kernel/Initializer/random_uniform:0
\
dense_6/bias:0dense_6/bias/Assigndense_6/bias/read:02 dense_6/bias/Initializer/zeros:0
m
dense_7/kernel:0dense_7/kernel/Assigndense_7/kernel/read:02+dense_7/kernel/Initializer/random_uniform:0
\
dense_7/bias:0dense_7/bias/Assigndense_7/bias/read:02 dense_7/bias/Initializer/zeros:0
l
Adam_1/iterations:0Adam_1/iterations/AssignAdam_1/iterations/read:02!Adam_1/iterations/initial_value:0
L
Adam_1/lr:0Adam_1/lr/AssignAdam_1/lr/read:02Adam_1/lr/initial_value:0
\
Adam_1/beta_1:0Adam_1/beta_1/AssignAdam_1/beta_1/read:02Adam_1/beta_1/initial_value:0
\
Adam_1/beta_2:0Adam_1/beta_2/AssignAdam_1/beta_2/read:02Adam_1/beta_2/initial_value:0
X
Adam_1/decay:0Adam_1/decay/AssignAdam_1/decay/read:02Adam_1/decay/initial_value:0
y
training_1/Adam/Variable:0training_1/Adam/Variable/Assigntraining_1/Adam/Variable/read:02training_1/Adam/Const_2:0

training_1/Adam/Variable_1:0!training_1/Adam/Variable_1/Assign!training_1/Adam/Variable_1/read:02training_1/Adam/Const_3:0

training_1/Adam/Variable_2:0!training_1/Adam/Variable_2/Assign!training_1/Adam/Variable_2/read:02training_1/Adam/Const_4:0

training_1/Adam/Variable_3:0!training_1/Adam/Variable_3/Assign!training_1/Adam/Variable_3/read:02training_1/Adam/Const_5:0

training_1/Adam/Variable_4:0!training_1/Adam/Variable_4/Assign!training_1/Adam/Variable_4/read:02training_1/Adam/Const_6:0

training_1/Adam/Variable_5:0!training_1/Adam/Variable_5/Assign!training_1/Adam/Variable_5/read:02training_1/Adam/Const_7:0

training_1/Adam/Variable_6:0!training_1/Adam/Variable_6/Assign!training_1/Adam/Variable_6/read:02training_1/Adam/Const_8:0

training_1/Adam/Variable_7:0!training_1/Adam/Variable_7/Assign!training_1/Adam/Variable_7/read:02training_1/Adam/Const_9:0

training_1/Adam/Variable_8:0!training_1/Adam/Variable_8/Assign!training_1/Adam/Variable_8/read:02training_1/Adam/Const_10:0

training_1/Adam/Variable_9:0!training_1/Adam/Variable_9/Assign!training_1/Adam/Variable_9/read:02training_1/Adam/Const_11:0

training_1/Adam/Variable_10:0"training_1/Adam/Variable_10/Assign"training_1/Adam/Variable_10/read:02training_1/Adam/Const_12:0

training_1/Adam/Variable_11:0"training_1/Adam/Variable_11/Assign"training_1/Adam/Variable_11/read:02training_1/Adam/Const_13:0

training_1/Adam/Variable_12:0"training_1/Adam/Variable_12/Assign"training_1/Adam/Variable_12/read:02training_1/Adam/Const_14:0

training_1/Adam/Variable_13:0"training_1/Adam/Variable_13/Assign"training_1/Adam/Variable_13/read:02training_1/Adam/Const_15:0

training_1/Adam/Variable_14:0"training_1/Adam/Variable_14/Assign"training_1/Adam/Variable_14/read:02training_1/Adam/Const_16:0

training_1/Adam/Variable_15:0"training_1/Adam/Variable_15/Assign"training_1/Adam/Variable_15/read:02training_1/Adam/Const_17:0"ľ3
	variables§3¤3
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
training/Adam/Variable_15:0 training/Adam/Variable_15/Assign training/Adam/Variable_15/read:02training/Adam/Const_17:0
m
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02+dense_4/kernel/Initializer/random_uniform:0
\
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02 dense_4/bias/Initializer/zeros:0
m
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:02+dense_5/kernel/Initializer/random_uniform:0
\
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:02 dense_5/bias/Initializer/zeros:0
m
dense_6/kernel:0dense_6/kernel/Assigndense_6/kernel/read:02+dense_6/kernel/Initializer/random_uniform:0
\
dense_6/bias:0dense_6/bias/Assigndense_6/bias/read:02 dense_6/bias/Initializer/zeros:0
m
dense_7/kernel:0dense_7/kernel/Assigndense_7/kernel/read:02+dense_7/kernel/Initializer/random_uniform:0
\
dense_7/bias:0dense_7/bias/Assigndense_7/bias/read:02 dense_7/bias/Initializer/zeros:0
l
Adam_1/iterations:0Adam_1/iterations/AssignAdam_1/iterations/read:02!Adam_1/iterations/initial_value:0
L
Adam_1/lr:0Adam_1/lr/AssignAdam_1/lr/read:02Adam_1/lr/initial_value:0
\
Adam_1/beta_1:0Adam_1/beta_1/AssignAdam_1/beta_1/read:02Adam_1/beta_1/initial_value:0
\
Adam_1/beta_2:0Adam_1/beta_2/AssignAdam_1/beta_2/read:02Adam_1/beta_2/initial_value:0
X
Adam_1/decay:0Adam_1/decay/AssignAdam_1/decay/read:02Adam_1/decay/initial_value:0
y
training_1/Adam/Variable:0training_1/Adam/Variable/Assigntraining_1/Adam/Variable/read:02training_1/Adam/Const_2:0

training_1/Adam/Variable_1:0!training_1/Adam/Variable_1/Assign!training_1/Adam/Variable_1/read:02training_1/Adam/Const_3:0

training_1/Adam/Variable_2:0!training_1/Adam/Variable_2/Assign!training_1/Adam/Variable_2/read:02training_1/Adam/Const_4:0

training_1/Adam/Variable_3:0!training_1/Adam/Variable_3/Assign!training_1/Adam/Variable_3/read:02training_1/Adam/Const_5:0

training_1/Adam/Variable_4:0!training_1/Adam/Variable_4/Assign!training_1/Adam/Variable_4/read:02training_1/Adam/Const_6:0

training_1/Adam/Variable_5:0!training_1/Adam/Variable_5/Assign!training_1/Adam/Variable_5/read:02training_1/Adam/Const_7:0

training_1/Adam/Variable_6:0!training_1/Adam/Variable_6/Assign!training_1/Adam/Variable_6/read:02training_1/Adam/Const_8:0

training_1/Adam/Variable_7:0!training_1/Adam/Variable_7/Assign!training_1/Adam/Variable_7/read:02training_1/Adam/Const_9:0

training_1/Adam/Variable_8:0!training_1/Adam/Variable_8/Assign!training_1/Adam/Variable_8/read:02training_1/Adam/Const_10:0

training_1/Adam/Variable_9:0!training_1/Adam/Variable_9/Assign!training_1/Adam/Variable_9/read:02training_1/Adam/Const_11:0

training_1/Adam/Variable_10:0"training_1/Adam/Variable_10/Assign"training_1/Adam/Variable_10/read:02training_1/Adam/Const_12:0

training_1/Adam/Variable_11:0"training_1/Adam/Variable_11/Assign"training_1/Adam/Variable_11/read:02training_1/Adam/Const_13:0

training_1/Adam/Variable_12:0"training_1/Adam/Variable_12/Assign"training_1/Adam/Variable_12/read:02training_1/Adam/Const_14:0

training_1/Adam/Variable_13:0"training_1/Adam/Variable_13/Assign"training_1/Adam/Variable_13/read:02training_1/Adam/Const_15:0

training_1/Adam/Variable_14:0"training_1/Adam/Variable_14/Assign"training_1/Adam/Variable_14/read:02training_1/Adam/Const_16:0

training_1/Adam/Variable_15:0"training_1/Adam/Variable_15/Assign"training_1/Adam/Variable_15/read:02training_1/Adam/Const_17:0înIĹ       çÎř	CČw2Ń×A*


accĽ°ő>ŞÖĐ       ŁK"	Éw2Ń×A*

lossďş?ă´kL       	uĘw2Ń×A*

val_acck<??Úźă       ČÁ	KËw2Ń×A*

val_lossYÎ?ô/       ń(	LĂM&2Ń×A*


acc ?1o$Í       Ř-	ëÄM&2Ń×A*

lossů§?Šd$F       `/ß#	"ĆM&2Ń×A*

val_acc×,?Ílnä       ŮÜ2	GÇM&2Ń×A*

val_lossËýq?Ç|Ró       ń(	@022Ń×A*


acc(č4?Ř.ĹŻ       Ř-	 022Ń×A*

loss/[?íËŃG       `/ß#	~022Ń×A*

val_accf$??ĎJr>       ŮÜ2	Y022Ń×A*

val_lossKD?íUjˇ       ń(	`O>2Ń×A*


acc˛=D?zTng       Ř-	_bO>2Ń×A*

losséK5?.%Ţ       `/ß#	scO>2Ń×A*

val_accÔĺH?hśŤ       ŮÜ2	ńdO>2Ń×A*

val_lossę§&?˛1äm       ń(	 ÔI2Ń×A*


accš$P?ÄĘ+       Ř-	wĄÔI2Ń×A*

lossŹ?0&       `/ß#	s˘ÔI2Ń×A*

val_accĽĽJ?/úV       ŮÜ2	UŁÔI2Ń×A*

val_lossR?Ů7š       ń(	çŞďU2Ń×A*


accËČW?"       Ř-	iŹďU2Ń×A*

lossG ?E       `/ß#	m­ďU2Ń×A*

val_accşZ?(K&Â       ŮÜ2	mŽďU2Ń×A*

val_loss$žů>űŘô       ń(	Zö)b2Ń×A*


acc}^?łQç       Ř-	đ÷)b2Ń×A*

lossA;Ü>3Ňô       `/ß#	ů)b2Ń×A*

val_acc(^?ĺMĘč       ŮÜ2	7ú)b2Ń×A*

val_loss-Ô>3Ý_ä       ń(	tďgn2Ń×A*


accöÂc?ĽA,Ţ       Ř-	Ôđgn2Ń×A*

lossőż>Ś°o       `/ß#	ťńgn2Ń×A*

val_accá}e?É.        ŮÜ2	ňgn2Ń×A*

val_lossďŕł>˘ë6v       ń(	=dGz2Ń×A*


accôîg?ľD+f       Ř-	eGz2Ń×A*

loss`Ś>ÇĄ(U       `/ß#	sfGz2Ń×A*

val_accÂDj?"äĺŢ       ŮÜ2	MgGz2Ń×A*

val_lossÍo>\ßďÇ       ń(	-2Ń×A	*


accęk?ä:mĐ       Ř-	-2Ń×A	*

loss	I>ĘőË       `/ß#	-2Ń×A	*

val_acc$}l?ÓT E       ŮÜ2	-2Ń×A	*

val_loss˙_>ěcÉJ       ń(	ř8ň2Ń×A
*


acc(n?NţGŰ       Ř-	K:ň2Ń×A
*

lossW>oZM       `/ß#	*;ň2Ń×A
*

val_accÝo?/îş       ŮÜ2	<ň2Ń×A
*

val_lossL@u>łš|ň       ń(	UM2Ń×A*


accép?ĚÚ§_       Ř-	ßO2Ń×A*

lossrb>0Ţ       `/ß#	÷V2Ń×A*

val_accąwq?7ú˘       ŮÜ2	u[2Ń×A*

val_loss
5_>dřn       ń(	Ź2Ń×A*


accŕůr?DńÂ       Ř-	ŮŹ2Ń×A*

lossQ8G>źzí       `/ß#	Ź2Ń×A*

val_accY-v?=-       ŮÜ2	%Ź2Ń×A*

val_lossôď.>6tö       ń(	Ü˝¸2Ń×A*


accĎt?Ş˝ô       Ř-	<ż¸2Ń×A*

loss<0>š.Ęă       `/ß#	Ŕ¸2Ń×A*

val_accć
v?ę       ŮÜ2	ýŔ¸2Ń×A*

val_loss;Ů>f­B       ń(	Ĺ2Ń×A*


accBv?Ľí÷Ł       Ř-	Ĺ2Ń×A*

lossíp>Řŕ´       `/ß#	gĹ2Ń×A*

val_accY-v?×żˇĐ       ŮÜ2	ŠĹ2Ń×A*

val_lossaq#>H=o       ń(	e4 Ń2Ń×A*


accˇx?ńs       Ř-	ô6 Ń2Ń×A*

loss$>ţX(       `/ß#	Ź8 Ń2Ń×A*

val_accY-v?5/       ŮÜ2	`: Ń2Ń×A*

val_losst>ÍĄ%       ń(	đÁÝ2Ń×A*


accĺ$y?b       Ř-	3ÁÝ2Ń×A*

loss|`ń=e       `/ß#	çÁÝ2Ń×A*

val_accýGz?.ş3¸       ŮÜ2	ÁÝ2Ń×A*

val_lossxńŕ=Ć|       ń(	˝#ę2Ń×A*


accťIz?č×I       Ř-	#ę2Ń×A*

loss~çÔ=ćśą       `/ß#	É#ę2Ń×A*

val_accúŹy?Xąt       ŮÜ2	~#ę2Ń×A*

val_lossľé=^Ŕ	U       ń(	öyö2Ń×A*


accťűz?ŕ˛âQ       Ř-	ŕö2Ń×A*

lossđŐź=ÍY       `/ß#	 ö2Ń×A*

val_accÄ6z?ČŞW,       ŮÜ2	 ö2Ń×A*

val_lossÓuĘ=Ěíp       ń(	R|?3Ń×A*


accČÂ{?Lš
       Ř-	Ş}?3Ń×A*

lossE$¨=H§šŢ       `/ß#	~?3Ń×A*

val_accâz?\V(       ŮÜ2	?3Ń×A*

val_lossÓwŔ=w*