package tensorlin.core.session

import tensorlin.core.TFOperation
import tensorlin.core.TFSession
import tensorlin.core.TFTensor
import tensorlin.core.graph.Graph
import tensorlin.core.status.Status.Companion.statusValidated
import kotlinx.cinterop.*
import tensorflow.*

class Session(val graph: Graph) {
    private val inputs = mutableListOf<TF_Output>()
    private val inputValues = mutableListOf<TFTensor>()
    private var outputs = mutableListOf<TF_Output>()
    private val outputValues = mutableListOf<TFTensor?>()
    private val targets = listOf<TFOperation>()
    @ExperimentalUnsignedTypes
    private var tfSession: TFSession? = createNewSession()


    @ExperimentalUnsignedTypes
    private fun createNewSession(): TFSession {
        val options = TF_NewSessionOptions()
        val session = statusValidated { TF_NewSession(graph.tfGraph, options, it.tfStatus)!! }
        TF_DeleteSessionOptions(options)
        return session
    }

    private fun setInputsWithValues(inputsWithValues: List<Pair<TFOperation, TFTensor>>) {
        clearInputValues()
        clearInputs()
        for ((input, inputValue) in inputsWithValues) {
            this.inputs.add(nativeHeap.alloc<TF_Output>().apply { oper = input; index = 0 })
            inputValues.add(inputValue)
        }
    }

    private fun setOutputs(outputs: List<TFOperation>) {
        clearOutputValues()
        clearOutputs()
        this.outputs = outputs.map { nativeHeap.alloc<TF_Output>().apply { oper = it; index = 0 } }.toMutableList()
    }

    private fun clearInputValues() {
        for (inputValue in inputValues) {
            TF_DeleteTensor(inputValue)
        }

        inputValues.clear()
    }

    private fun clearOutputValues() {
        for (outputValue in outputValues) {
            if (outputValue != null)
                TF_DeleteTensor(outputValue)
        }
        outputValues.clear()
    }

    private fun clearOutputs() {
        this.outputs.forEach { nativeHeap.free(it) }
        this.outputs.clear()
    }

    private fun clearInputs() {
        this.inputs.forEach { nativeHeap.free(it) }
        this.inputs.clear()
    }

    @ExperimentalUnsignedTypes
    fun dispose() {
        clearInputValues()
        clearOutputValues()
        clearInputs()
        clearOutputs()

        if (tfSession != null) {
            statusValidated { TF_CloseSession(tfSession, it.tfStatus) }
            statusValidated { TF_DeleteSession(tfSession, it.tfStatus) }
            tfSession = null
        }
    }

    @ExperimentalUnsignedTypes
    operator fun invoke(outputs: List<TFOperation>, inputsWithValues: List<Pair<TFOperation, TFTensor>> = listOf()): List<TFTensor?> {
        setInputsWithValues(inputsWithValues)
        setOutputs(outputs)

        return invoke()
    }

    operator fun invoke(output: TFOperation, inputsWithValues: List<Pair<TFOperation, TFTensor>> = listOf()) =
        invoke(listOf(output), inputsWithValues).single()!!

    @ExperimentalUnsignedTypes
    operator fun invoke(): List<TFTensor?> {
        if (inputs.size != inputValues.size) {
            throw Error("Call SetInputs() before Run()")
        }
        clearOutputValues()

        val inputsCArray = if (inputs.any()) nativeHeap.allocArray<TF_Output>(inputs.size) else null

        inputs.forEachIndexed { i, input ->
            inputsCArray!![i].apply {
                oper = input.oper
                index = input.index
            }
        }

        val outputsCArray = if (outputs.any()) nativeHeap.allocArray<TF_Output>(outputs.size) else null

        outputs.forEachIndexed { i, output ->
            outputsCArray!![i].apply {
                oper = output.oper
                index = output.index
            }
        }

        memScoped {
            val outputValuesCArray = allocArrayOfPointersTo<TF_Tensor>(outputs.map { null })

            statusValidated {
                TF_SessionRun(tfSession, null,
                    inputsCArray, inputValues.toCValues(), inputs.size,
                    outputsCArray, outputValuesCArray, outputs.size,
                    targets.toCValues(), targets.size,
                    null, it.tfStatus)
            }

            for (index in outputs.indices) {
                outputValues.add(outputValuesCArray[index])
            }
        }

        clearInputValues()

        return outputValues
    }
}