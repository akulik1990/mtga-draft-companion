// One card's shield: v2's score, styled by pack rank. Grey stays quiet, bronze and silver are brushed metal,
// gold catches a sweeping light, fire and blue fire burn. Used by draft.qml.
import QtQuick
import QtQuick.Effects
import QtQuick.Particles
import QtQuick.Shapes

Item {
    id: s
    property string tier: "grey"  // grey | bronze | silver | gold | fire | bluefire
    property bool glow: false     // a clear pick: tops both Fit and Power
    property var score: null
    property int rank: 0

    readonly property bool hot: tier === "fire" || tier === "bluefire"
    readonly property bool metal: tier !== "grey"
    readonly property var look: ({
        grey:     { rim: ["#4a5058", "#2c3036", "#3a3f46", "#23272c"], edge: "#5a616a", plate: ["#1c2024", "#121518"], text: "#c9ced4" },
        bronze:   { rim: ["#f1c08f", "#8a4f24", "#d89a63", "#5e3416"], edge: "#c98a55", plate: ["#2a1d14", "#140e0a"], text: "#ffe2c4" },
        silver:   { rim: ["#ffffff", "#8e9aa8", "#e9f0f8", "#5f6b78"], edge: "#dfe8f2", plate: ["#20262d", "#101418"], text: "#f4f8ff" },
        gold:     { rim: ["#fff6c2", "#b9850f", "#ffe27a", "#7a5306"], edge: "#ffd54f", plate: ["#2b2410", "#140f05"], text: "#fff4cf" },
        fire:     { rim: ["#fff0a0", "#e0441c", "#ffb347", "#8a1a06"], edge: "#ff8a2a", plate: ["#3a140a", "#1a0704"], text: "#fff1d6" },
        bluefire: { rim: ["#e8fbff", "#1d63d8", "#7fd8ff", "#0b2f80"], edge: "#4fc3ff", plate: ["#0d2240", "#060f1f"], text: "#e6f7ff" }
    })[tier]

    // Bigger and brighter the higher it ranks.
    scale: ({ grey: 0.86, bronze: 0.95, silver: 1.0, gold: 1.1, fire: 1.14, bluefire: 1.16 })[tier]
    opacity: tier === "grey" ? 0.82 : 1.0
    transformOrigin: Item.Center

    // Shield outline in relative coordinates, inset by m pixels.
    function outline(m) {
        const X = r => m + r * (width - 2 * m), Y = r => m + r * (height - 2 * m);
        return `M ${X(.5)} ${Y(0)} L ${X(1)} ${Y(.14)} L ${X(1)} ${Y(.5)} Q ${X(1)} ${Y(.84)} ${X(.5)} ${Y(1)} Q ${X(0)} ${Y(.84)} ${X(0)} ${Y(.5)} L ${X(0)} ${Y(.14)} Z`;
    }

    // Flames: particles rising off the top of the shield, behind it.
    ParticleSystem {
        id: flames
        running: s.hot
        anchors.fill: parent
    }
    ImageParticle {
        system: flames; groups: ["flame"]
        source: "qrc:///particleresources/glowdot.png"
        color: s.tier === "bluefire" ? "#3aa8ff" : "#ff6a1a"
        colorVariation: 0.1
        alpha: 0.85
    }
    ImageParticle {
        system: flames; groups: ["core"]
        source: "qrc:///particleresources/glowdot.png"
        color: s.tier === "bluefire" ? "#d8f6ff" : "#ffe08a"
        alpha: 0.95
    }
    ImageParticle {
        system: flames; groups: ["ember"]
        source: "qrc:///particleresources/fuzzydot.png"
        color: s.tier === "bluefire" ? "#9fe4ff" : "#ffc36b"
        colorVariation: 0.2
    }
    Emitter {
        system: flames; group: "flame"; enabled: s.hot
        x: s.width * 0.08; y: s.height * 0.02; width: s.width * 0.84; height: s.height * 0.22
        emitRate: 160; lifeSpan: 750; lifeSpanVariation: 250
        size: s.width * 0.7; sizeVariation: s.width * 0.2; endSize: s.width * 0.12
        velocity: AngleDirection { angle: 270; angleVariation: 12; magnitude: s.height * 1.3; magnitudeVariation: s.height * 0.4 }
        acceleration: PointDirection { y: -s.height * 0.6; xVariation: s.width * 0.4 }
    }
    Emitter {
        system: flames; group: "core"; enabled: s.hot
        x: s.width * 0.25; y: s.height * 0.06; width: s.width * 0.5; height: s.height * 0.12
        emitRate: 90; lifeSpan: 500; size: s.width * 0.45; endSize: s.width * 0.08
        velocity: AngleDirection { angle: 270; angleVariation: 8; magnitude: s.height * 0.9 }
    }
    Emitter {
        system: flames; group: "ember"; enabled: s.hot
        x: s.width * 0.2; y: s.height * 0.05; width: s.width * 0.6; height: s.height * 0.1
        emitRate: 14; lifeSpan: 1500; lifeSpanVariation: 400; size: s.width * 0.07; sizeVariation: s.width * 0.03
        velocity: AngleDirection { angle: 270; angleVariation: 30; magnitude: s.height * 0.9; magnitudeVariation: s.height * 0.4 }
        acceleration: PointDirection { xVariation: s.width * 0.6 }
    }

    // Rim and plate, with a glow for a clear pick (pulsing) or a fire tier (steady, brighter).
    Item {
        id: body
        anchors.fill: parent
        property real pulse: 0.6
        SequentialAnimation on pulse {
            running: s.glow && !s.hot; loops: Animation.Infinite
            NumberAnimation { from: 0.35; to: 0.95; duration: 1000; easing.type: Easing.InOutSine }
            NumberAnimation { from: 0.95; to: 0.35; duration: 1000; easing.type: Easing.InOutSine }
        }
        layer.enabled: s.glow || s.hot
        layer.effect: MultiEffect {
            shadowEnabled: true; shadowColor: s.look.edge; shadowBlur: 1.0
            shadowOpacity: s.hot ? 1.0 : body.pulse
            shadowHorizontalOffset: 0; shadowVerticalOffset: 0
            blurMax: 48; autoPaddingEnabled: true
        }
        Shape {
            anchors.fill: parent
            preferredRendererType: Shape.CurveRenderer
            ShapePath {  // rim: banded gradient reads as brushed metal
                strokeColor: s.look.edge; strokeWidth: s.metal ? 1.5 : 1
                fillGradient: LinearGradient {
                    x1: 0; y1: 0; x2: s.width; y2: s.height
                    GradientStop { position: 0.0; color: s.look.rim[0] }
                    GradientStop { position: 0.42; color: s.look.rim[1] }
                    GradientStop { position: 0.58; color: s.look.rim[2] }
                    GradientStop { position: 1.0; color: s.look.rim[3] }
                }
                PathSvg { path: s.outline(0) }
            }
            ShapePath {  // bevel: a thin light line just inside the rim
                strokeColor: s.metal ? Qt.rgba(1, 1, 1, 0.45) : "transparent"; strokeWidth: 1
                fillColor: "transparent"
                PathSvg { path: s.outline(s.width * 0.06) }
            }
            ShapePath {  // plate
                strokeColor: Qt.rgba(0, 0, 0, 0.5); strokeWidth: 1
                fillGradient: RadialGradient {
                    centerX: s.width / 2; centerY: s.height * 0.42; centerRadius: s.width * 0.55
                    focalX: s.width / 2; focalY: s.height * 0.3
                    GradientStop { position: 0; color: s.look.plate[0] }
                    GradientStop { position: 1; color: s.look.plate[1] }
                }
                PathSvg { path: s.outline(s.width * 0.13) }
            }
        }
    }

    // Gold and fire: a light band sweeping across the shield now and then.
    Item {
        id: sheen
        anchors.fill: parent
        visible: s.tier === "gold" || s.hot
        layer.enabled: visible
        layer.effect: MultiEffect { maskEnabled: true; maskSource: mask; maskThresholdMin: 0.5 }
        Rectangle {
            id: band
            width: s.width * 0.32; height: s.height * 1.8; y: -s.height * 0.4
            rotation: 22
            gradient: Gradient {
                orientation: Gradient.Horizontal
                GradientStop { position: 0; color: "transparent" }
                GradientStop { position: 0.5; color: Qt.rgba(1, 1, 1, 0.55) }
                GradientStop { position: 1; color: "transparent" }
            }
            SequentialAnimation on x {
                running: sheen.visible; loops: Animation.Infinite
                NumberAnimation { from: -s.width * 0.8; to: s.width * 1.4; duration: 900; easing.type: Easing.InOutQuad }
                PauseAnimation { duration: s.hot ? 1400 : 2600 }
            }
        }
    }
    Shape {
        id: mask
        anchors.fill: parent
        visible: false
        layer.enabled: true
        ShapePath { strokeColor: "transparent"; fillColor: "white"; PathSvg { path: s.outline(0) } }
    }

    Text {
        anchors.centerIn: parent
        anchors.verticalCenterOffset: -s.height * 0.03
        text: s.score ? Math.round(s.score) : "–"
        color: s.look.text
        style: s.metal ? Text.Raised : Text.Normal; styleColor: Qt.rgba(0, 0, 0, 0.6)
        font.pixelSize: s.width * 0.36; font.bold: true
    }
    Rectangle {  // pack rank chip
        visible: s.rank > 0
        width: s.width * 0.38; height: width; radius: width / 2
        x: s.width * 0.78; y: -height * 0.15
        color: "#15181c"
        border.color: s.metal ? s.look.edge : "#80868e"
        border.width: s.rank === 1 ? 2 : 1
        Text {
            anchors.centerIn: parent
            text: s.rank
            color: s.metal ? s.look.edge : "#d0d4d8"
            font.pixelSize: parent.width * 0.55; font.bold: true
        }
    }
}
