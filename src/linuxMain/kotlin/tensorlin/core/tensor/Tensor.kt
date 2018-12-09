package tensorlin.core.tensor

import tensorlin.core.SizeT
import tensorlin.core.TFTensor
import kotlinx.cinterop.*
import tensorflow.*


class Tensor<T>(
    val value: T,
    val arg0: TF_DataType = TF_INT32,
    val dims: CValuesRef<LongVarOf<Long>>? = null,
    val num_dims: Int = 0,
    val len: SizeT = IntVar.size.convert(),
    val deallocator_arg: CValuesRef<*>? = null,
    val deallocator: CPointer<CFunction<(COpaquePointer?, SizeT, COpaquePointer?) -> Unit>> = staticCFunction { dataToFree, _, _ ->
        nativeHeap.free(
            dataToFree!!.reinterpret<IntVar>()
        )
    }
) {
    lateinit var tfTensor: TFTensor

    init {
        when (value) {
            is Int -> {
                val data: CArrayPointer<IntVar> = nativeHeap.allocArray(1)
                data[0] = value
                tfTensor = TF_NewTensor(
                    data = data,
                    arg0 = arg0,
                    dims = dims,
                    num_dims = num_dims,
                    len = len,
                    deallocator = deallocator,
                    deallocator_arg = deallocator_arg
                )!!
            }
            is List<*> -> {
                //TODO "We not ready for this!"
            }
        }

    }

}