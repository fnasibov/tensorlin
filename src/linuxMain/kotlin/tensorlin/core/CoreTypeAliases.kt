package tensorlin.core

import kotlinx.cinterop.CPointer
import platform.posix.size_t
import tensorflow.TF_Tensor
import tensorflow.TF_Status
import tensorflow.TF_Operation
import tensorflow.TF_Session

public typealias SizeT = size_t
public typealias TFTensor = CPointer<TF_Tensor>
public typealias TFStatus = CPointer<TF_Status>
public typealias TFOperation = CPointer<TF_Operation>
public typealias TFSession = CPointer<TF_Session>

