package tensorlin

import tensorlin.core.SizeT
import tensorlin.core.TFTensor
import tensorlin.core.graph.Graph
import tensorlin.core.tensor.Tensor
import kotlinx.cinterop.*
import tensorflow.TF_Version

@ExperimentalUnsignedTypes
val TFTensor.scalarIntValue: Int
    get() {
        if (tensorflow.TF_INT32 != tensorflow.TF_TensorType(this) || IntVar.size.convert<SizeT>() != tensorflow.TF_TensorByteSize(
                this
            )
        ) {
            throw kotlin.Error("Tensor is not of type int.")
        }
        if (0 != tensorflow.TF_NumDims(this)) {
            throw kotlin.Error("Tensor is not scalar.")
        }

        return tensorflow.TF_TensorData(this)!!.reinterpret<IntVar>().pointed.value
    }

@ExperimentalUnsignedTypes
fun main(args: Array<String>) {
    println("TensorFlow version: " + TF_Version()?.toKString())

    var a = 2
    var b = 3

    if (args.size >= 2) {
        a = args[0].toInt()
        b = args[1].toInt()
    }

    val result = Graph().run {
        val input = intInput()

        withSession {
            invoke(
                input + constant(a),
                inputsWithValues = listOf(input to Tensor(b).tfTensor)
            ).scalarIntValue
        }
    }
    println("$a + $b is $result")
}

