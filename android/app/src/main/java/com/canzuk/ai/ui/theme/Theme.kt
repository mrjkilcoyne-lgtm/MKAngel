package com.canzuk.ai.ui.theme

import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = CanzukRed,
    onPrimary = CanzukWhite,
    secondary = CanzukBlue,
    onSecondary = CanzukWhite,
    tertiary = GoldAccent,
    background = NavySurface,
    surface = NavyLight,
    onBackground = TextPrimary,
    onSurface = TextPrimary,
    error = ErrorRed,
    outline = TextDim,
)

private val LightColorScheme = lightColorScheme(
    primary = CanzukRed,
    onPrimary = CanzukWhite,
    secondary = CanzukBlue,
    onSecondary = CanzukWhite,
    tertiary = GoldAccent,
    background = CanzukWhite,
    surface = Color(0xFFF5F5F8),
    onBackground = NavySurface,
    onSurface = NavySurface,
    error = ErrorRed,
    outline = Silver,
)

@Composable
fun CanzukTheme(
    darkTheme: Boolean = true,
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme,
        typography = CanzukTypography,
        content = content,
    )
}
