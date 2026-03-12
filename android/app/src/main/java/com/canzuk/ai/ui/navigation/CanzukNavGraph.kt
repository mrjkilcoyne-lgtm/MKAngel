package com.canzuk.ai.ui.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.canzuk.ai.engine.EngineViewModel
import com.canzuk.ai.ui.chat.ChatScreen
import com.canzuk.ai.ui.chat.ChatViewModel
import com.canzuk.ai.ui.screens.CanvasScreen
import com.canzuk.ai.ui.screens.CodeScreen
import com.canzuk.ai.ui.screens.DocumentScreen
import com.canzuk.ai.ui.screens.MoreScreen

enum class Screen(val route: String) {
    Chat("chat"),
    Document("document"),
    Code("code"),
    Canvas("canvas"),
    More("more"),
}

@Composable
fun CanzukNavGraph(navController: NavHostController) {
    val engine: EngineViewModel = viewModel()
    val chatViewModel = remember { ChatViewModel(engine) }

    NavHost(navController = navController, startDestination = Screen.Chat.route) {
        composable(Screen.Chat.route) { ChatScreen(chatViewModel) }
        composable(Screen.Document.route) { DocumentScreen() }
        composable(Screen.Code.route) { CodeScreen() }
        composable(Screen.Canvas.route) { CanvasScreen() }
        composable(Screen.More.route) { MoreScreen() }
    }
}
