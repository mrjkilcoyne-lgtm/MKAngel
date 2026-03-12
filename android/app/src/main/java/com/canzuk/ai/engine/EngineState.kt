package com.canzuk.ai.engine

data class EngineState(
    val ready: Boolean = false,
    val loading: Boolean = true,
    val error: String? = null,
    val domains: List<String> = emptyList(),
    val parameters: Int = 0,
    val grammars: Int = 0,
    val strangeLoops: Int = 0,
)
