package com.canzuk.ai.engine

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.chaquo.python.Python
import com.chaquo.python.PyObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class EngineViewModel : ViewModel() {

    private val _state = MutableStateFlow(EngineState())
    val state: StateFlow<EngineState> = _state.asStateFlow()

    private var bridge: PyObject? = null

    init {
        viewModelScope.launch {
            initialise()
        }
    }

    private suspend fun initialise() {
        _state.update { it.copy(loading = true) }
        try {
            withContext(Dispatchers.IO) {
                val py = Python.getInstance()
                val module = py.getModule("glm.bridge")
                bridge = module.callAttr("CanzukBridge")
            }
            // Get introspection info
            val info = withContext(Dispatchers.IO) {
                bridge?.callAttr("introspect")?.toJava(Map::class.java) as? Map<*, *>
            }
            _state.update {
                it.copy(
                    ready = true,
                    loading = false,
                    domains = (info?.get("domains") as? List<*>)?.map { d -> d.toString() } ?: emptyList(),
                    parameters = (info?.get("parameters") as? Number)?.toInt() ?: 303754,
                    grammars = (info?.get("grammars") as? Number)?.toInt() ?: 0,
                    strangeLoops = (info?.get("strange_loops") as? Number)?.toInt() ?: 0,
                )
            }
        } catch (e: Exception) {
            _state.update { it.copy(loading = false, error = e.message) }
        }
    }

    /**
     * Stream response tokens from the grammar engine.
     * Returns a Flow that emits tokens as they are generated.
     */
    fun stream(input: String): Flow<String> = flow {
        val iterator = withContext(Dispatchers.IO) {
            bridge?.callAttr("stream", input)
        }
        // Chaquopy returns a PyObject wrapping the Python iterator
        iterator?.asList()?.forEach { token ->
            emit(token.toString())
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Process input and return complete response.
     */
    suspend fun process(input: String): String = withContext(Dispatchers.IO) {
        bridge?.callAttr("process", input)?.toString() ?: "Engine not ready."
    }
}
