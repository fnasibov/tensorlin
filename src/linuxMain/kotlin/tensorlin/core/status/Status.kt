package tensorlin.core.status

import kotlinx.cinterop.toKString
import tensorflow.*

class Status {

    var tfStatus = TF_NewStatus()!!

    @ExperimentalUnsignedTypes
    val isOk: Boolean
        get() = TF_GetCode(tfStatus) == TF_OK

    val errorMessage: String
        get() = TF_Message(tfStatus)!!.toKString()


    @ExperimentalUnsignedTypes
    fun validate() {
        try {
            if (!isOk) {
                throw Error("Status is not OK: $errorMessage")
            }
        } finally {
            delete()
        }
    }

    private fun delete() = TF_DeleteStatus(tfStatus)

    companion object {

        @ExperimentalUnsignedTypes
        inline fun <T> statusValidated(block: (Status) -> T): T {
            val status = Status()
            val result = block(status)
            status.validate()
            return result
        }
    }

}