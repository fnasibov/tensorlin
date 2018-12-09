package tensorlin.core.graph

import tensorlin.core.TFOperation
import tensorlin.core.session.Session
import tensorlin.core.status.Status.Companion.statusValidated
import tensorlin.core.tensor.Tensor
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import tensorflow.*

class Graph {
    val tfGraph = TF_NewGraph()!!

    @ExperimentalUnsignedTypes
    inline fun operation(type: String, name: String, initDescription: (CPointer<TF_OperationDescription>) -> Unit): TFOperation {
        val description = TF_NewOperation(tfGraph, type, name)!!
        initDescription(description)
        return statusValidated { TF_FinishOperation(description, it.tfStatus)!! }
    }

    @ExperimentalUnsignedTypes
    fun constant(value: Int, name: String = "scalarIntConstant") = operation("Const", name) { description ->
        statusValidated { TF_SetAttrTensor(description, "value", Tensor(value).tfTensor, it.tfStatus) }
        TF_SetAttrType(description, "dtype", TF_INT32)
    }

    @ExperimentalUnsignedTypes
    fun intInput(name: String = "input") = operation("Placeholder", name) { description ->
        TF_SetAttrType(description, "dtype", TF_INT32)
    }

    @ExperimentalUnsignedTypes
    fun add(left: TFOperation, right: TFOperation, name: String = "add") = memScoped {
        val inputs = allocArray<TF_Output>(2)
        inputs[0].apply { oper = left; index = 0 }
        inputs[1].apply { oper = right; index = 0 }

        operation("AddN", name) { description ->
            TF_AddInputList(description, inputs, 2)
        }
    }

    // TODO set unique operation names
    @ExperimentalUnsignedTypes
    operator fun TFOperation.plus(right: TFOperation) = add(this, right)

    @ExperimentalUnsignedTypes
    inline fun <T> withSession(block: Session.() -> T): T {
        val session = Session(this)
        try {
            return session.block()
        } finally {
            session.dispose()
        }
    }
}