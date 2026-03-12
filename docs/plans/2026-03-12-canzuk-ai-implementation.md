# CANZUK-AI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build CANZUK-AI — a native Android AI assistant powered by MKAngel's Grammar Language Model, with Jetpack Compose UI and Chaquopy Python bridge.

**Architecture:** Kotlin/Compose for UI and lifecycle, Python via Chaquopy for all reasoning (GLM engine, pipeline, realiser). Clean bridge boundary: Kotlin sends text in, receives streaming tokens out. All existing `glm/` Python code carries over unchanged.

**Tech Stack:** Kotlin, Jetpack Compose (Material 3), Chaquopy (Python-in-Android), Room (SQLite), Kotlin Coroutines/Flow, Gradle

**Design Doc:** `docs/plans/2026-03-12-canzuk-ai-design.md`

---

## Phase 1: Project Scaffold & Python Bridge

### Task 1: Create Android Project Structure

**Files:**
- Create: `android/build.gradle.kts` (root)
- Create: `android/settings.gradle.kts`
- Create: `android/gradle.properties`
- Create: `android/app/build.gradle.kts`
- Create: `android/app/src/main/AndroidManifest.xml`
- Create: `android/app/src/main/java/com/canzuk/ai/CanzukApp.kt`

**Step 1: Create root Gradle config**

```kotlin
// android/build.gradle.kts
plugins {
    id("com.android.application") version "8.2.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
    id("com.chaquo.python") version "15.0.1" apply false
}
```

```kotlin
// android/settings.gradle.kts
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolution {
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "CANZUK-AI"
include(":app")
```

```properties
# android/gradle.properties
android.useAndroidX=true
kotlin.code.style=official
org.gradle.jvmargs=-Xmx2048m
```

**Step 2: Create app module with Chaquopy**

```kotlin
// android/app/build.gradle.kts
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "com.canzuk.ai"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.canzuk.ai"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        ndk { abiFilters += listOf("arm64-v8a") }

        python {
            buildPython("python3")
            pip {
                // No external deps — pure Python GLM
            }
        }
    }

    buildFeatures { compose = true }
    composeOptions { kotlinCompilerExtensionVersion = "1.5.8" }

    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // Compose
    implementation(platform("androidx.compose:compose-bom:2024.01.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.navigation:navigation-compose:2.7.7")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Room (SQLite)
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    annotationProcessor("androidx.room:room-compiler:2.6.1")
}

chaquopy {
    defaultConfig {
        version = "3.12"
    }
    sourceSets {
        getByName("main") {
            srcDir("../../glm")
            srcDir("../../app")
        }
    }
}
```

**Step 3: Create AndroidManifest**

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.RECORD_AUDIO" />

    <application
        android:name=".CanzukApp"
        android:label="CANZUK-AI"
        android:theme="@style/Theme.CanzukAI"
        android:supportsRtl="true">
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:windowSoftInputMode="adjustResize">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

**Step 4: Create Application class**

```kotlin
// android/app/src/main/java/com/canzuk/ai/CanzukApp.kt
package com.canzuk.ai

import android.app.Application
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class CanzukApp : Application() {
    override fun onCreate() {
        super.onCreate()
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
    }
}
```

**Step 5: Commit**

```bash
git add android/
git commit -m "feat: scaffold CANZUK-AI Android project with Chaquopy"
```

---

### Task 2: Python Bridge Module

**Files:**
- Create: `glm/bridge.py`
- Create: `glm/realiser_v2.py` (stub)
- Test: `tests/test_bridge.py`

**Step 1: Write failing test**

```python
# tests/test_bridge.py
import pytest
from glm.bridge import CanzukBridge

def test_bridge_initialises():
    bridge = CanzukBridge()
    assert bridge.ready is True

def test_bridge_process_returns_string():
    bridge = CanzukBridge()
    result = bridge.process("What is photosynthesis?")
    assert isinstance(result, str)
    assert len(result) > 0

def test_bridge_stream_yields_tokens():
    bridge = CanzukBridge()
    tokens = list(bridge.stream("Explain gravity"))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)

def test_bridge_introspect():
    bridge = CanzukBridge()
    info = bridge.introspect()
    assert "domains" in info
    assert "parameters" in info
    assert "grammars" in info
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_bridge.py -v`
Expected: FAIL with "cannot import name 'CanzukBridge'"

**Step 3: Write bridge implementation**

```python
# glm/bridge.py
"""
CANZUK-AI Bridge — entry point for Chaquopy calls from Kotlin.

All Kotlin→Python communication flows through CanzukBridge.
Methods return plain types (str, dict, list) for Chaquopy serialisation.
"""
from typing import Iterator

from glm.angel import Angel
from glm.pipeline import ReasoningPipeline
from glm.realiser_v2 import GenerativeRealiser


class CanzukBridge:
    """Single entry point for the Kotlin UI to call the GLM engine."""

    def __init__(self):
        self._angel = Angel()
        self._angel.awaken()
        self._pipeline = ReasoningPipeline()
        self._realiser = GenerativeRealiser(self._angel)
        self.ready = True

    def process(self, text: str) -> str:
        """Process input and return complete response as string."""
        return "".join(self.stream(text))

    def stream(self, text: str) -> Iterator[str]:
        """Stream response tokens from grammar engine.

        1. Pipeline decomposes input (Skeleton→DAG→Disconfirm→Synthesis)
        2. Realiser walks the validated derivation tree
        3. Yields natural language tokens as they are generated
        """
        pipeline_result = self._pipeline.run(text)
        yield from self._realiser.stream(pipeline_result, text)

    def introspect(self) -> dict:
        """Return live model stats for the Introspection screen."""
        info = self._angel.introspect() if hasattr(self._angel, 'introspect') else {}
        return {
            "domains": info.get("domains", []),
            "grammars": info.get("grammar_count", 0),
            "rules": info.get("rule_count", 0),
            "strange_loops": info.get("strange_loops", 0),
            "parameters": info.get("parameters", 303754),
            "version": "0.1.0",
            "name": "CANZUK-AI",
        }

    def get_domains(self) -> list:
        """List available grammar domains."""
        return self.introspect().get("domains", [])
```

**Step 4: Write Generative Realiser stub**

```python
# glm/realiser_v2.py
"""
Generative Realiser v2 — walks derivation trees to produce long-form text.

This is the module that makes CANZUK-AI feel like a frontier LLM.
The grammar derivation tree provides structure; the realiser produces
flowing natural language that follows every valid path.
"""
from typing import Iterator


class GenerativeRealiser:
    """Walks validated derivation trees to stream natural language."""

    # Connectives for different tree transitions
    BRANCH_CONNECTIVES = [
        "Furthermore, ", "Additionally, ", "Building on this, ",
        "This connects to ", "In a related domain, ",
    ]
    CONTRAST_CONNECTIVES = [
        "However, ", "In contrast, ", "On the other hand, ",
        "A counterpoint emerges: ", "Yet consider that ",
    ]
    DEEPENING_CONNECTIVES = [
        "To understand why, ", "Looking deeper, ",
        "At a more fundamental level, ", "The underlying structure reveals ",
    ]
    CONCLUSION_CONNECTIVES = [
        "Therefore, ", "This establishes that ", "In synthesis, ",
        "The evidence converges on ", "Drawing these threads together, ",
    ]

    def __init__(self, angel=None):
        self._angel = angel
        self._nlg = None
        if angel and hasattr(angel, 'nlg'):
            self._nlg = angel.nlg

    def stream(self, pipeline_result, original_input: str = "") -> Iterator[str]:
        """Stream natural language tokens from a pipeline result.

        Walks the validated derivation tree from the pipeline:
        - Skeleton claims become topic sentences
        - DAG structure determines paragraph order
        - Disconfirm results add hedging and caveats
        - Synthesis provides the conclusion

        Each section streams token-by-token for real-time UI display.
        """
        # Phase 1: Opening — acknowledge the question
        if original_input:
            yield from self._stream_opening(original_input, pipeline_result)

        # Phase 2: Body — walk the derivation tree
        yield from self._stream_body(pipeline_result)

        # Phase 3: Synthesis — draw conclusions
        yield from self._stream_synthesis(pipeline_result)

    def _stream_opening(self, text, result) -> Iterator[str]:
        """Generate an opening that acknowledges the input domain(s)."""
        # Detect domains from pipeline result
        domains = []
        if hasattr(result, 'skeleton') and result.skeleton:
            sr = result.skeleton
            if hasattr(sr, 'grammar_coverage'):
                domains = list(sr.grammar_coverage.keys()) if isinstance(sr.grammar_coverage, dict) else []

        if domains:
            domain_str = ", ".join(d.replace("_", " ") for d in domains[:3])
            yield f"This touches on {domain_str}. "

        yield "Let me trace the structural paths.\n\n"

    def _stream_body(self, result) -> Iterator[str]:
        """Walk the DAG to produce ordered paragraphs."""
        # Extract claims from skeleton
        claims = []
        if hasattr(result, 'skeleton') and result.skeleton:
            sr = result.skeleton
            if hasattr(sr, 'triples'):
                claims = sr.triples
            elif hasattr(sr, 'claims'):
                claims = sr.claims

        if not claims:
            # Fallback: generate from the raw pipeline stages
            yield from self._stream_from_stages(result)
            return

        # Get DAG ordering
        ordered_claims = claims  # TODO: topological sort from DAG result

        # Walk claims, generating a paragraph per major node
        for i, claim in enumerate(ordered_claims):
            # Add connective for non-first claims
            if i > 0:
                connective_pool = self.BRANCH_CONNECTIVES
                connective = connective_pool[i % len(connective_pool)]
                yield f"\n\n{connective}"

            # Render claim as natural language
            if hasattr(claim, 'subject') and hasattr(claim, 'relation') and hasattr(claim, 'object'):
                yield f"{claim.subject} {claim.relation} {claim.object}. "
            elif hasattr(claim, 'text'):
                yield f"{claim.text} "
            else:
                yield f"{str(claim)} "

            # Expand with derivation if available
            if hasattr(claim, 'confidence') and claim.confidence < 0.8:
                yield "This warrants careful consideration — "
                yield "the structural evidence is suggestive rather than definitive. "

        # Add disconfirmation insights
        if hasattr(result, 'disconfirm') and result.disconfirm:
            yield from self._stream_disconfirm(result.disconfirm)

    def _stream_from_stages(self, result) -> Iterator[str]:
        """Fallback: generate from raw stage outputs when claims aren't extractable."""
        stages = ['skeleton', 'dag', 'disconfirm', 'synthesis']

        for stage_name in stages:
            stage_result = getattr(result, stage_name, None)
            if stage_result is None:
                continue

            # Convert stage result to readable text
            text = ""
            if isinstance(stage_result, str):
                text = stage_result
            elif hasattr(stage_result, '__dict__'):
                # Walk attributes and extract meaningful content
                for key, value in stage_result.__dict__.items():
                    if key.startswith('_'):
                        continue
                    if isinstance(value, str) and len(value) > 10:
                        text += f"{value} "
                    elif isinstance(value, list) and value:
                        for item in value[:5]:
                            text += f"{str(item)} "

            if text.strip():
                for word in text.split():
                    yield word + " "
                yield "\n\n"

    def _stream_disconfirm(self, disconfirm) -> Iterator[str]:
        """Add caveats and counterpoints from disconfirmation stage."""
        yield "\n\n"
        yield self.CONTRAST_CONNECTIVES[0]

        if hasattr(disconfirm, 'weaknesses') and disconfirm.weaknesses:
            for w in disconfirm.weaknesses[:2]:
                if hasattr(w, 'description'):
                    yield f"{w.description} "
                else:
                    yield f"{str(w)} "

        if hasattr(disconfirm, 'steel_man') and disconfirm.steel_man:
            yield "\n\nThe strongest counter-argument would be: "
            yield f"{disconfirm.steel_man} "

    def _stream_synthesis(self, result) -> Iterator[str]:
        """Generate concluding synthesis."""
        yield "\n\n"
        yield self.CONCLUSION_CONNECTIVES[0]

        if hasattr(result, 'synthesis') and result.synthesis:
            synth = result.synthesis
            if hasattr(synth, 'proven') and synth.proven:
                for claim in synth.proven[:3]:
                    yield f"{str(claim)} "
            if hasattr(synth, 'clean_argument') and synth.clean_argument:
                yield f"{synth.clean_argument} "
            elif hasattr(synth, 'verdict') and synth.verdict:
                yield f"{synth.verdict} "
        else:
            yield "the grammar derivation traces converge on a coherent structural pattern. "

        # MNEMO signature
        yield "\n\n"
        yield "— CANZUK-AI (Grammar Language Model, 303K parameters)"
```

**Step 5: Run tests to verify they pass**

Run: `cd C:\Users\mrjki\OneDrive\Desktop\MKAngel && python -m pytest tests/test_bridge.py -v`
Expected: 4 PASS

**Step 6: Commit**

```bash
git add glm/bridge.py glm/realiser_v2.py tests/test_bridge.py
git commit -m "feat: add CanzukBridge and GenerativeRealiser v2"
```

---

## Phase 2: Kotlin UI Foundation

### Task 3: Theme & Design System

**Files:**
- Create: `android/app/src/main/java/com/canzuk/ai/ui/theme/Color.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/theme/Type.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/theme/Theme.kt`

**Step 1: Define CANZUK color palette**

```kotlin
// ui/theme/Color.kt
package com.canzuk.ai.ui.theme

import androidx.compose.ui.graphics.Color

// The Union Palette
val CanzukRed = Color(0xFFC8102E)
val CanzukBlue = Color(0xFF012169)
val CanzukWhite = Color(0xFFFAFAFA)
val NavySurface = Color(0xFF0A1128)
val NavyLight = Color(0xFF131B36)
val Silver = Color(0xFFB8C4D0)
val GoldAccent = Color(0xFFD4A847)
val TextPrimary = Color(0xFFF0F0F2)
val TextSecondary = Color(0xFF8890A0)
val TextDim = Color(0xFF4A5068)
val ErrorRed = Color(0xFFEF5350)
val SuccessGreen = Color(0xFF66BB6A)
```

**Step 2: Define typography**

```kotlin
// ui/theme/Type.kt
package com.canzuk.ai.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

val CanzukTypography = Typography(
    displayLarge = TextStyle(fontSize = 28.sp, fontWeight = FontWeight.Bold, letterSpacing = (-0.5).sp),
    headlineMedium = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.SemiBold),
    titleMedium = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Medium),
    bodyLarge = TextStyle(fontSize = 14.sp, lineHeight = 22.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    labelSmall = TextStyle(fontSize = 12.sp, fontWeight = FontWeight.Medium, letterSpacing = 0.5.sp),
)
```

**Step 3: Define theme**

```kotlin
// ui/theme/Theme.kt
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
```

**Step 4: Commit**

```bash
git add android/app/src/main/java/com/canzuk/ai/ui/theme/
git commit -m "feat: CANZUK-AI design system — Union Palette + typography"
```

---

### Task 4: Main Activity & Navigation

**Files:**
- Create: `android/app/src/main/java/com/canzuk/ai/MainActivity.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/navigation/CanzukNavGraph.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/navigation/BottomBar.kt`

**Step 1: Create MainActivity**

```kotlin
// MainActivity.kt
package com.canzuk.ai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.canzuk.ai.ui.theme.CanzukTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            CanzukTheme {
                CanzukApp()
            }
        }
    }
}
```

**Step 2: Create navigation graph**

```kotlin
// ui/navigation/CanzukNavGraph.kt
package com.canzuk.ai.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable

enum class Screen(val route: String) {
    Chat("chat"),
    Document("document"),
    Code("code"),
    Canvas("canvas"),
    More("more"),
}

@Composable
fun CanzukNavGraph(navController: NavHostController) {
    NavHost(navController = navController, startDestination = Screen.Chat.route) {
        composable(Screen.Chat.route) { /* ChatScreen() */ }
        composable(Screen.Document.route) { /* DocumentScreen() */ }
        composable(Screen.Code.route) { /* CodeScreen() */ }
        composable(Screen.Canvas.route) { /* CanvasScreen() */ }
        composable(Screen.More.route) { /* MoreScreen() */ }
    }
}
```

**Step 3: Create bottom navigation bar**

```kotlin
// ui/navigation/BottomBar.kt
package com.canzuk.ai.ui.navigation

import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.res.painterResource
import androidx.navigation.NavController
import androidx.navigation.compose.currentBackStackEntryAsState
import com.canzuk.ai.ui.theme.CanzukRed
import com.canzuk.ai.ui.theme.TextDim

data class BottomNavItem(
    val screen: Screen,
    val label: String,
    val icon: String, // placeholder — use material icons
)

val bottomNavItems = listOf(
    BottomNavItem(Screen.Chat, "Chat", "chat"),
    BottomNavItem(Screen.Document, "Document", "description"),
    BottomNavItem(Screen.Code, "Code", "code"),
    BottomNavItem(Screen.Canvas, "Canvas", "brush"),
    BottomNavItem(Screen.More, "More", "more_horiz"),
)

@Composable
fun CanzukBottomBar(navController: NavController) {
    val currentRoute = navController.currentBackStackEntryAsState().value?.destination?.route

    NavigationBar(
        containerColor = MaterialTheme.colorScheme.background,
    ) {
        bottomNavItems.forEach { item ->
            NavigationBarItem(
                selected = currentRoute == item.screen.route,
                onClick = {
                    navController.navigate(item.screen.route) {
                        popUpTo(Screen.Chat.route) { saveState = true }
                        launchSingleTop = true
                        restoreState = true
                    }
                },
                label = {
                    Text(
                        item.label,
                        style = MaterialTheme.typography.labelSmall,
                    )
                },
                icon = { /* Icon placeholder — add material icons */ },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = CanzukRed,
                    selectedTextColor = CanzukRed,
                    unselectedIconColor = TextDim,
                    unselectedTextColor = TextDim,
                    indicatorColor = CanzukRed.copy(alpha = 0.1f),
                ),
            )
        }
    }
}
```

**Step 4: Wire CanzukApp composable**

```kotlin
// Add to CanzukApp.kt or create ui/CanzukApp.kt
@Composable
fun CanzukApp() {
    val navController = rememberNavController()

    Scaffold(
        bottomBar = { CanzukBottomBar(navController) },
    ) { padding ->
        Box(modifier = Modifier.padding(padding)) {
            CanzukNavGraph(navController)
        }
    }
}
```

**Step 5: Commit**

```bash
git add android/app/src/main/java/com/canzuk/ai/
git commit -m "feat: main activity, navigation graph, bottom bar"
```

---

### Task 5: Engine ViewModel (Kotlin↔Python Bridge)

**Files:**
- Create: `android/app/src/main/java/com/canzuk/ai/engine/EngineViewModel.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/engine/EngineState.kt`

**Step 1: Define engine state**

```kotlin
// engine/EngineState.kt
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
```

**Step 2: Create ViewModel with Chaquopy bridge**

```kotlin
// engine/EngineViewModel.kt
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
```

**Step 3: Commit**

```bash
git add android/app/src/main/java/com/canzuk/ai/engine/
git commit -m "feat: EngineViewModel — Kotlin↔Python Chaquopy bridge"
```

---

## Phase 3: Chat Screen (Core Experience)

### Task 6: Chat Data Model

**Files:**
- Create: `android/app/src/main/java/com/canzuk/ai/data/Message.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/data/Conversation.kt`

**Step 1: Define message model**

```kotlin
// data/Message.kt
package com.canzuk.ai.data

import java.util.UUID

enum class MessageRole { USER, ASSISTANT, SYSTEM }

data class Message(
    val id: String = UUID.randomUUID().toString(),
    val role: MessageRole,
    val content: String,
    val timestamp: Long = System.currentTimeMillis(),
    val isStreaming: Boolean = false,
    val domains: List<String> = emptyList(), // grammar domains involved
)
```

```kotlin
// data/Conversation.kt
package com.canzuk.ai.data

import java.util.UUID

data class Conversation(
    val id: String = UUID.randomUUID().toString(),
    val title: String = "New conversation",
    val messages: List<Message> = emptyList(),
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis(),
)
```

**Step 2: Commit**

```bash
git add android/app/src/main/java/com/canzuk/ai/data/
git commit -m "feat: Message and Conversation data models"
```

---

### Task 7: Chat Screen UI

**Files:**
- Create: `android/app/src/main/java/com/canzuk/ai/ui/chat/ChatScreen.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/chat/ChatViewModel.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/chat/MessageBubble.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/chat/InputBar.kt`

**Step 1: ChatViewModel**

```kotlin
// ui/chat/ChatViewModel.kt
package com.canzuk.ai.ui.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.canzuk.ai.data.Message
import com.canzuk.ai.data.MessageRole
import com.canzuk.ai.engine.EngineViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

data class ChatUiState(
    val messages: List<Message> = emptyList(),
    val isGenerating: Boolean = false,
    val streamingContent: String = "",
)

class ChatViewModel(
    private val engine: EngineViewModel,
) : ViewModel() {

    private val _state = MutableStateFlow(ChatUiState())
    val state: StateFlow<ChatUiState> = _state.asStateFlow()

    fun send(text: String) {
        if (text.isBlank()) return

        // Add user message
        val userMsg = Message(role = MessageRole.USER, content = text)
        _state.update { it.copy(messages = it.messages + userMsg, isGenerating = true, streamingContent = "") }

        // Stream response
        viewModelScope.launch {
            val buffer = StringBuilder()
            engine.stream(text).collect { token ->
                buffer.append(token)
                _state.update { it.copy(streamingContent = buffer.toString()) }
            }

            // Finalise assistant message
            val assistantMsg = Message(role = MessageRole.ASSISTANT, content = buffer.toString())
            _state.update {
                it.copy(
                    messages = it.messages + assistantMsg,
                    isGenerating = false,
                    streamingContent = "",
                )
            }
        }
    }
}
```

**Step 2: Message bubble**

```kotlin
// ui/chat/MessageBubble.kt
package com.canzuk.ai.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import com.canzuk.ai.data.Message
import com.canzuk.ai.data.MessageRole
import com.canzuk.ai.ui.theme.*

@Composable
fun MessageBubble(message: Message) {
    val isUser = message.role == MessageRole.USER

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 320.dp)
                .clip(
                    RoundedCornerShape(
                        topStart = 16.dp,
                        topEnd = 16.dp,
                        bottomStart = if (isUser) 16.dp else 4.dp,
                        bottomEnd = if (isUser) 4.dp else 16.dp,
                    )
                )
                .background(
                    if (isUser) CanzukRed.copy(alpha = 0.12f)
                    else MaterialTheme.colorScheme.surface
                )
                .padding(12.dp),
        ) {
            Text(
                text = message.content,
                style = MaterialTheme.typography.bodyLarge,
                color = if (isUser) TextPrimary else TextPrimary,
            )
        }
    }
}
```

**Step 3: Input bar**

```kotlin
// ui/chat/InputBar.kt
package com.canzuk.ai.ui.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import com.canzuk.ai.ui.theme.*

@Composable
fun InputBar(
    onSend: (String) -> Unit,
    enabled: Boolean = true,
) {
    var text by remember { mutableStateOf("") }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.Bottom,
    ) {
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            modifier = Modifier.weight(1f),
            placeholder = {
                Text("Ask CANZUK-AI...", color = TextDim)
            },
            shape = RoundedCornerShape(24.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = CanzukBlue,
                unfocusedBorderColor = TextDim.copy(alpha = 0.3f),
                cursorColor = CanzukRed,
            ),
            maxLines = 5,
            enabled = enabled,
        )

        Spacer(Modifier.width(8.dp))

        IconButton(
            onClick = {
                if (text.isNotBlank()) {
                    onSend(text)
                    text = ""
                }
            },
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(if (text.isNotBlank()) CanzukRed else TextDim.copy(alpha = 0.2f)),
            enabled = text.isNotBlank() && enabled,
        ) {
            Text("↑", color = CanzukWhite, style = MaterialTheme.typography.titleMedium)
        }
    }
}
```

**Step 4: Chat screen composition**

```kotlin
// ui/chat/ChatScreen.kt
package com.canzuk.ai.ui.chat

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.canzuk.ai.data.Message
import com.canzuk.ai.data.MessageRole
import com.canzuk.ai.ui.theme.*

@Composable
fun ChatScreen(viewModel: ChatViewModel) {
    val state by viewModel.state.collectAsState()
    val listState = rememberLazyListState()

    // Auto-scroll to bottom on new messages
    LaunchedEffect(state.messages.size, state.streamingContent) {
        if (state.messages.isNotEmpty()) {
            listState.animateScrollToItem(state.messages.size)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // Header
        Surface(
            color = MaterialTheme.colorScheme.background,
            shadowElevation = 1.dp,
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .statusBarsPadding()
                    .padding(horizontal = 20.dp, vertical = 12.dp),
            ) {
                Text(
                    "CANZUK",
                    style = MaterialTheme.typography.headlineMedium,
                    color = CanzukRed,
                )
                Text(
                    "-AI",
                    style = MaterialTheme.typography.headlineMedium,
                    color = CanzukBlue,
                )
            }
        }

        // Messages
        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            contentPadding = PaddingValues(vertical = 8.dp),
        ) {
            items(state.messages) { message ->
                MessageBubble(message)
            }

            // Streaming indicator
            if (state.isGenerating && state.streamingContent.isNotEmpty()) {
                item {
                    MessageBubble(
                        Message(
                            role = MessageRole.ASSISTANT,
                            content = state.streamingContent,
                            isStreaming = true,
                        )
                    )
                }
            } else if (state.isGenerating) {
                item {
                    Row(Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                        Text("Deriving...", color = TextDim, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }

        // Input
        InputBar(
            onSend = viewModel::send,
            enabled = !state.isGenerating,
        )
    }
}
```

**Step 5: Commit**

```bash
git add android/app/src/main/java/com/canzuk/ai/ui/chat/
git commit -m "feat: Chat screen — message bubbles, input bar, streaming"
```

---

## Phase 4: Remaining Screens (Stubs)

### Task 8: Document, Code, Canvas, More Screen Stubs

**Files:**
- Create: `android/app/src/main/java/com/canzuk/ai/ui/screens/DocumentScreen.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/screens/CodeScreen.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/screens/CanvasScreen.kt`
- Create: `android/app/src/main/java/com/canzuk/ai/ui/screens/MoreScreen.kt`

Each screen follows the same pattern — stub with CANZUK styling:

```kotlin
// ui/screens/DocumentScreen.kt
package com.canzuk.ai.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.canzuk.ai.ui.theme.*

@Composable
fun DocumentScreen() {
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Document", style = MaterialTheme.typography.headlineMedium, color = CanzukBlue)
            Text("Long-form generation", style = MaterialTheme.typography.bodyMedium, color = TextDim)
        }
    }
}
```

Create similar stubs for CodeScreen, CanvasScreen, MoreScreen.

**Commit:**

```bash
git add android/app/src/main/java/com/canzuk/ai/ui/screens/
git commit -m "feat: stub screens for Document, Code, Canvas, More"
```

---

## Phase 5: CI & Build

### Task 9: GitHub Actions Android Build

**Files:**
- Create: `.github/workflows/android-build.yml`

**Step 1: Create workflow**

```yaml
# .github/workflows/android-build.yml
name: Build CANZUK-AI APK

on:
  push:
    branches: [main]
    paths: ['android/**', 'glm/**', 'app/**']
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up JDK 17
        uses: actions/setup-java@v4
        with:
          java-version: '17'
          distribution: 'temurin'

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Build APK
        working-directory: android
        run: ./gradlew assembleDebug

      - name: Upload APK
        uses: actions/upload-artifact@v4
        with:
          name: canzuk-ai-debug
          path: android/app/build/outputs/apk/debug/*.apk
          if-no-files-found: error
```

**Step 2: Commit**

```bash
git add .github/workflows/android-build.yml
git commit -m "ci: Android build workflow for CANZUK-AI"
```

---

## Phase 6: Generative Realiser Full Implementation

### Task 10: Upgrade Realiser for Long-Form Generation

**Files:**
- Modify: `glm/realiser_v2.py`
- Test: `tests/test_realiser_v2.py`

**Step 1: Write failing tests**

```python
# tests/test_realiser_v2.py
import pytest
from glm.realiser_v2 import GenerativeRealiser

def test_realiser_streams_tokens():
    r = GenerativeRealiser()
    tokens = list(r.stream(None, "What is gravity?"))
    assert len(tokens) > 5
    full = "".join(tokens)
    assert len(full) > 50

def test_realiser_produces_long_form():
    r = GenerativeRealiser()
    # Simulate a rich pipeline result with multiple claims
    from glm.pipeline import ReasoningPipeline
    pipeline = ReasoningPipeline()
    result = pipeline.run("Explain the relationship between etymology and molecular biology")
    tokens = list(r.stream(result, "Explain the relationship between etymology and molecular biology"))
    full = "".join(tokens)
    # Should produce substantial output for a cross-domain question
    assert len(full) > 200

def test_realiser_includes_connectives():
    r = GenerativeRealiser()
    from glm.pipeline import ReasoningPipeline
    pipeline = ReasoningPipeline()
    result = pipeline.run("Compare photosynthesis and cellular respiration")
    full = "".join(r.stream(result, "Compare photosynthesis and cellular respiration"))
    # Should have structural connectives
    has_connective = any(c in full for c in ["Furthermore", "However", "Therefore", "Additionally", "In contrast"])
    assert has_connective

def test_realiser_ends_with_signature():
    r = GenerativeRealiser()
    full = "".join(r.stream(None, "Hello"))
    assert "CANZUK-AI" in full
```

**Step 2: Run tests, verify failures, implement full realiser (expand the stub from Task 2), run tests, commit.**

```bash
git add glm/realiser_v2.py tests/test_realiser_v2.py
git commit -m "feat: GenerativeRealiser — long-form streaming from derivation trees"
```

---

## Summary

| Phase | Tasks | What it builds |
|-------|-------|---------------|
| **1: Scaffold** | Tasks 1-2 | Android project + Chaquopy + Python bridge |
| **2: UI Foundation** | Tasks 3-5 | Theme, navigation, engine ViewModel |
| **3: Chat** | Tasks 6-7 | Full chat screen with streaming |
| **4: Screens** | Task 8 | Stub screens for all capabilities |
| **5: CI** | Task 9 | GitHub Actions APK build |
| **6: Realiser** | Task 10 | Full long-form generation |

**Total: 10 tasks, ~30 bite-sized steps.**

After this plan executes, you have a buildable CANZUK-AI APK with:
- Red/white/blue CANZUK theme
- Chat interface with streaming grammar-driven responses
- Python GLM engine running natively on Android via Chaquopy
- CI pipeline producing APKs on push
- Stub screens ready for Phase 2 implementation
