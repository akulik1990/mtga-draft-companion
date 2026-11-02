// Badges over Arena's draft cards: a transparent, click-through layer-shell surface (niri overlay layer).
// Run with overlay/run.sh. OVERLAY_TEST=1 draws 14 dummy badges to check alignment without a draft.
// The API sorts the pack the way Arena displays it. Arena has two draft layouts; one screen pixel tells them apart.
// ponytail: assumes Arena fills the output. Grids measured on 2560x1440, 2026-11-02.
import QtQuick
import QtQuick.Shapes
import QtQuick.Effects
import Quickshell
import Quickshell.Io
import Quickshell.Wayland

ShellRoot {
    id: root

    // Card grids as fractions of the screen. side: deck list on the right, 5 per row; bottom: deck under the pack, 8 per row.
    readonly property var layouts: ({
        side: { x0: 0.145, dx: 0.1025, cardW: 0.09, y0: 0.1707, dy: 0.2489, cardH: 0.227, perRow: 5 },
        bottom: { x0: 0.1715, dx: 0.0835, cardW: 0.069, y0: 0.1636, dy: 0.1858, cardH: 0.17, perRow: 8 }
    })
    // The bottom layout's deck panel covers this point in a flat (9,16,16); the side layout shows the pack or backdrop there.
    readonly property var probe: [0.5, 0.85]
    property string layoutName: "side"
    readonly property var g: layouts[layoutName]
    property bool showNames: false // card name on each badge, to spot a wrong order at a glance

    property string screenName: Quickshell.env("OVERLAY_SCREEN") || "DP-1"
    property string api: (Quickshell.env("OVERLAY_API") || "http://localhost:17173") + "/api/overlay"
    property bool test: Quickshell.env("OVERLAY_TEST") === "1"
    property var data: null
    property bool arenaFocused: false
    property real px: -1  // pointer, in screen pixels (from overlay/pointer.py)
    property real py: -1
    // Arena opens a large preview over the pack while the pointer is on a card; the badges step aside.
    readonly property bool overCard: {
        if (!data || !win.screen) return false;
        const fx = (px - win.screen.x) / win.width, fy = (py - win.screen.y) / win.height;
        const col = Math.floor((fx - g.x0) / g.dx), row = Math.floor((fy - g.y0) / g.dy);
        return col >= 0 && col < g.perRow && row >= 0 && fx - g.x0 - col * g.dx <= g.cardW
            && fy - g.y0 - row * g.dy <= g.cardH && row * g.perRow + col < (data.cards || []).length + 1;  // + the basic land Arena shows but the log omits
    }

    function poll() {
        const x = new XMLHttpRequest();
        x.onreadystatechange = () => {
            if (x.readyState === XMLHttpRequest.DONE)
                root.data = x.status === 200 ? JSON.parse(x.responseText) : null;
        };
        x.open("GET", api);
        x.send();
    }

    function dummy() {
        const cards = [];
        for (let i = 0; i < 14; i++)
            cards.push({ name: "card " + (i + 1), rank: i + 1, score: 60 - i, orders: "Fit #" + (i + 1) + " · Power #" + (14 - i), upstream: i + 1 });
        return { cards: cards, branch: "Stay: card 1 (not card 2)" };
    }

    Timer {
        interval: 1000; running: true; repeat: true; triggeredOnStart: true
        onTriggered: {
            if (root.test) { root.data = root.dummy(); root.arenaFocused = true; return; }
            root.poll();
            if (!focus.running) focus.running = true;
            if (root.arenaFocused && !pixel.running) pixel.running = true;
        }
    }

    // Shown while Arena is the active window on our output, focused or not (the other monitor may have focus).
    Process {
        id: focus
        command: ["sh", "-c", "niri msg --json workspaces; niri msg --json windows"]
        stdout: StdioCollector {
            onStreamFinished: {
                try {
                    const [spaces, windows] = text.trim().split("\n").map(JSON.parse);
                    const ws = spaces.find(w => w.output === root.screenName && w.is_active);
                    const win = windows.find(w => w.id === (ws || {}).active_window_id);
                    root.arenaFocused = !!win && win.app_id === "steam_app_2141910";
                } catch (e) { root.arenaFocused = false; }
            }
        }
    }

    Process {
        id: pixel
        command: ["sh", "-c", "grim -t ppm -g \"" + Math.round(win.screen.x + root.probe[0] * win.width) + "," + Math.round(win.screen.y + root.probe[1] * win.height) + " 1x1\" - | tail -c 3 | od -An -tu1"]
        stdout: StdioCollector {
            onStreamFinished: {
                const v = text.trim().split(/\s+/).map(Number);
                if (v.length === 3)
                    root.layoutName = v.every((c, i) => Math.abs(c - [9, 16, 16][i]) <= 6) ? "bottom" : "side";
            }
        }
    }

    Process {
        running: true
        command: ["python3", Qt.resolvedUrl("pointer.py").toString().replace("file://", "")]
        stdout: SplitParser {
            onRead: line => {
                const [x, y] = line.split(" ").map(Number);
                root.px = x; root.py = y;
            }
        }
    }

    PanelWindow {
        id: win
        screen: Quickshell.screens.find(s => s.name === root.screenName) ?? null
        anchors { top: true; bottom: true; left: true; right: true }
        exclusionMode: ExclusionMode.Ignore
        WlrLayershell.layer: WlrLayer.Overlay
        WlrLayershell.namespace: "mtga-draft-overlay"
        WlrLayershell.keyboardFocus: WlrKeyboardFocus.None
        color: "transparent"
        mask: Region {}  // empty input region: every click goes to Arena
        visible: root.arenaFocused && !!root.data && (root.data.cards || []).length > 0

        // Shield colour by v2's pack rank. v2's pick turns to fire when the card is a standout (taken by pick 2 on average,
        // or GIH WR >= mean + 1 SD) and blue fire when it is one of the set's top 20 by GIH WR (/api/overlay "heat").
        // It glows when it also tops both Fit and Power: best for your colours and strongest outright.
        readonly property var medals: ({
            bluefire: { hi: "#e6fbff", lo: "#1550d0", edge: "#4fc3ff", flame: true },
            fire: { hi: "#ffe066", lo: "#d42a10", edge: "#ff7a1a", flame: true },
            gold: { hi: "#fff1a8", lo: "#c9971c", edge: "#ffd54f" },
            silver: { hi: "#ffffff", lo: "#b4c8e0", edge: "#eaf2ff" },
            bronze: { hi: "#d9a27a", lo: "#6e3f1e", edge: "#b87a4b" },
            grey: { hi: "#3c4148", lo: "#1d2024", edge: "#50565e" }
        })
        function clear(c) { return c.rank === 1 && c.fit === 1 && c.power === 1; }
        function medal(c) {
            if (c.rank === 1) return medals[c.heat === 2 ? "bluefire" : c.heat === 1 ? "fire" : "gold"];
            return medals[["", "", "silver", "bronze"][c.rank] || "grey"];
        }
        readonly property int unit: Math.round(win.height * 0.017)

        readonly property var wubrg: ({ W: "#efe3b8", U: "#4a90d9", B: "#a58fcf", R: "#e0583a", G: "#3fa35c" })
        readonly property var pn: root.data && root.data.panel ? root.data.panel : null

        // Left panel: v2's pick and why, likeliest lanes, open colours, pool. Sits in the empty strip left of the pack.
        Rectangle {
            id: side
            visible: !!win.pn && !root.overCard  // Arena's preview can reach over it too
            x: win.width * 0.008
            y: win.height * root.g.y0
            width: win.width * (root.g.x0 - 0.018)
            height: col.implicitHeight + win.unit * 1.2
            radius: win.unit * 0.6
            color: "#d90e1215"
            border.color: "#30ffffff"

            Column {
                id: col
                x: win.unit * 0.6; y: win.unit * 0.6
                width: side.width - win.unit * 1.2
                spacing: win.unit * 0.35

                component Heading: Text { color: "#8f98a3"; font.pixelSize: win.unit * 0.62; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                component Line: Text { color: "#e8eaed"; font.pixelSize: win.unit * 0.72; width: col.width; wrapMode: Text.Wrap }

                Heading { text: "V2 TOP 3" }
                Repeater {
                    model: win.pn ? win.pn.top3 : []
                    Column {
                        required property var modelData
                        required property int index
                        readonly property var m: win.medals[["gold", "silver", "bronze"][index]]
                        width: col.width
                        spacing: 1
                        Row {
                            spacing: win.unit * 0.4
                            Rectangle {
                                width: win.unit * 1.1; height: width; radius: width / 2
                                anchors.verticalCenter: parent.verticalCenter
                                gradient: Gradient {
                                    GradientStop { position: 0; color: m.hi }
                                    GradientStop { position: 1; color: m.lo }
                                }
                                Text { anchors.centerIn: parent; text: index + 1; color: "#1b1407"; font.pixelSize: win.unit * 0.7; font.bold: true }
                            }
                            Text {
                                width: col.width - win.unit * 4.4; elide: Text.ElideRight
                                text: modelData.name
                                color: index === 0 ? "#ffd54f" : "#e8eaed"; font.pixelSize: win.unit * 0.82; font.bold: index === 0
                            }
                            Text {
                                width: win.unit * 2.4; horizontalAlignment: Text.AlignRight
                                text: modelData.score ? (index === 0 ? Number(modelData.score).toFixed(1)
                                      : (modelData.score - win.pn.top3[0].score).toFixed(1)) : "n/a"
                                color: "#c8cdd3"; font.pixelSize: win.unit * 0.72
                            }
                        }
                        Line { visible: !!modelData.why; text: modelData.why; leftPadding: win.unit * 1.5; color: "#9aa3ad"; font.pixelSize: win.unit * 0.62 }
                    }
                }
                Line {  // how top drafters split between v2's top three when #1 leads by this much
                    visible: !!win.pn && !!win.pn.close && !!win.pn.close.took
                    text: win.pn && win.pn.close && win.pn.close.took
                          ? (win.pn.close.gap < 1 ? "Close call. " : win.pn.close.gap < 4 ? "" : "Clear. ")
                            + "#1 leads by " + win.pn.close.gap + "; top drafters here take #1 " + Math.round(win.pn.close.took[0])
                            + "%, #2 " + Math.round(win.pn.close.took[1]) + "%, #3 " + Math.round(win.pn.close.took[2]) + "%"
                          : ""
                    color: win.pn && win.pn.close && win.pn.close.gap < 1 ? "#ffcc80" : "#9fd3a8"
                    font.pixelSize: win.unit * 0.66
                }
                Line {
                    visible: !!root.data && !!root.data.branch
                    text: root.data ? root.data.branch : ""
                    color: root.data && root.data.branch.startsWith("Switch") ? "#ffb347" : "#8f98a3"
                    font.pixelSize: win.unit * 0.62
                }

                Item { width: 1; height: win.unit * 0.3 }
                Heading { text: "YOUR LANES" }
                Repeater {
                    model: win.pn ? win.pn.lanes : []
                    Row {
                        required property var modelData
                        spacing: win.unit * 0.4
                        Text { text: modelData.lane; width: win.unit * 1.6; color: "#e8eaed"; font.pixelSize: win.unit * 0.72; font.bold: true }
                        Item {  // probability bar in the pair's two colours
                            width: col.width - win.unit * 5.2; height: win.unit * 0.55
                            anchors.verticalCenter: parent.verticalCenter
                            Rectangle { anchors.fill: parent; radius: height / 2; color: "#2a2f35" }
                            Row {
                                height: parent.height
                                Rectangle { width: parent.parent.width * modelData.p / win.pn.lanes[0].p / 2; height: parent.height; color: win.wubrg[modelData.lane[0]] }
                                Rectangle { width: parent.parent.width * modelData.p / win.pn.lanes[0].p / 2; height: parent.height; color: win.wubrg[modelData.lane[1]] }
                            }
                        }
                        Text { text: Math.round(modelData.p * 100) + "%"; width: win.unit * 1.6; horizontalAlignment: Text.AlignRight; color: "#c8cdd3"; font.pixelSize: win.unit * 0.65 }
                    }
                }
                Line {
                    visible: !!win.pn && win.pn.lanes.length > 0
                    text: win.pn ? "Pair win rates: " + win.pn.lanes.map(l => l.lane + " " + (l.wr ? Number(l.wr).toFixed(1) : "–")).join(" · ") : ""
                    color: "#8f98a3"; font.pixelSize: win.unit * 0.58
                }

                Item { width: 1; height: win.unit * 0.3 }
                Heading { text: "OPEN COLOURS (passing to you)" }
                Row {
                    id: sig
                    readonly property var vals: win.pn ? win.pn.signals : ({})
                    readonly property real peak: Math.max(0.01, ...Object.values(vals).map(Math.abs))
                    visible: Object.values(vals).some(v => v !== 0)
                    spacing: win.unit * 0.3
                    Repeater {
                        model: ["W", "U", "B", "R", "G"]
                        Column {
                            required property string modelData
                            readonly property real v: sig.vals[modelData] || 0
                            spacing: 2
                            Item {  // centred bar: up = open, down = cut
                                width: (col.width - win.unit * 1.2) / 5; height: win.unit * 2.4
                                Rectangle { y: parent.height / 2; width: parent.width; height: 1; color: "#40ffffff" }
                                Rectangle {
                                    width: parent.width * 0.7; x: parent.width * 0.15
                                    height: Math.abs(parent.parent.v) / sig.peak * parent.height / 2
                                    y: parent.parent.v >= 0 ? parent.height / 2 - height : parent.height / 2
                                    radius: 2; color: win.wubrg[modelData]; opacity: parent.parent.v >= 0 ? 1 : 0.45
                                }
                            }
                            Text { anchors.horizontalCenter: parent.horizontalCenter; text: modelData; color: win.wubrg[modelData]; font.pixelSize: win.unit * 0.65; font.bold: true }
                        }
                    }
                }
                Line { visible: !sig.visible; text: "No signal yet"; color: "#8f98a3"; font.pixelSize: win.unit * 0.65 }

                Item { width: 1; height: win.unit * 0.3 }
                Heading { text: win.pn ? "POOL · " + win.pn.pool.n + " CARDS" : "POOL" }
                Row {
                    spacing: win.unit * 0.5
                    Repeater {
                        model: ["W", "U", "B", "R", "G"]
                        Row {
                            required property string modelData
                            spacing: 3
                            Rectangle { width: win.unit * 0.6; height: width; radius: width / 2; color: win.wubrg[modelData]; anchors.verticalCenter: parent.verticalCenter }
                            Text { text: win.pn ? win.pn.pool.colours[modelData] : 0; color: "#e8eaed"; font.pixelSize: win.unit * 0.68 }
                        }
                    }
                }
                Row {  // curve: spells by mana value 1..6+
                    id: curve
                    readonly property var c: win.pn ? win.pn.pool.curve : [0, 0, 0, 0, 0, 0]
                    readonly property int peak: Math.max(1, ...c)
                    spacing: win.unit * 0.25
                    Repeater {
                        model: 6
                        Column {
                            required property int index
                            spacing: 1
                            Item {
                                width: (col.width - win.unit * 1.3) / 6; height: win.unit * 1.8
                                Rectangle {
                                    anchors.bottom: parent.bottom; width: parent.width; radius: 2
                                    height: Math.max(1, curve.c[index] / curve.peak * parent.height)
                                    color: curve.c[index] ? "#7f8c99" : "#2a2f35"
                                }
                                Text { anchors.horizontalCenter: parent.horizontalCenter; y: -height + parent.height - Math.max(1, curve.c[index] / curve.peak * parent.height); visible: curve.c[index] > 0; text: curve.c[index]; color: "#c8cdd3"; font.pixelSize: win.unit * 0.55 }
                            }
                            Text { anchors.horizontalCenter: parent.horizontalCenter; text: index === 5 ? "6+" : index + 1; color: "#8f98a3"; font.pixelSize: win.unit * 0.55 }
                        }
                    }
                }
            }
        }

        Repeater {
            model: root.data ? root.data.cards : []
            delegate: Item {
                id: card
                visible: !root.overCard
                required property var modelData
                required property int index
                readonly property int col: index % root.g.perRow
                readonly property int row: Math.floor(index / root.g.perRow)
                readonly property var t: win.medal(modelData)
                readonly property bool glow: win.clear(modelData) || !!t.flame
                readonly property bool flame: !!t.flame
                opacity: modelData.rank && modelData.rank <= 3 ? 1.0 : 0.8
                // Shield on the card's bottom edge, clear of art and rules text.
                width: win.width * root.g.cardW * 0.34
                height: width * 1.15
                x: win.width * (root.g.x0 + col * root.g.dx + root.g.cardW / 2) - width / 2
                y: win.height * (root.g.y0 + row * root.g.dy + root.g.cardH) - height * 0.45

                Repeater {  // flames for fire / blue fire, flickering behind the shield's top
                    model: card.flame ? [[-0.28, 0.75, 0], [0, 1.0, 140], [0.28, 0.75, 280]] : []
                    Shape {
                        id: fl
                        required property var modelData
                        readonly property real w: card.width * 0.42
                        readonly property real h: card.height * 0.62 * modelData[1]
                        x: card.width / 2 + modelData[0] * card.width - w / 2
                        y: -h * 0.55
                        width: w; height: h
                        transformOrigin: Item.Bottom
                        preferredRendererType: Shape.CurveRenderer
                        SequentialAnimation on scale {
                            loops: Animation.Infinite
                            PauseAnimation { duration: modelData[2] }
                            NumberAnimation { from: 0.85; to: 1.12; duration: 420; easing.type: Easing.InOutQuad }
                            NumberAnimation { from: 1.12; to: 0.85; duration: 420; easing.type: Easing.InOutQuad }
                        }
                        ShapePath {
                            strokeColor: "transparent"
                            fillGradient: LinearGradient {
                                x1: 0; y1: 0; x2: 0; y2: fl.h
                                GradientStop { position: 0; color: "transparent" }
                                GradientStop { position: 0.35; color: card.t.edge }
                                GradientStop { position: 1; color: card.t.lo }
                            }
                            PathSvg { path: `M ${fl.w/2} 0 Q ${fl.w*0.95} ${fl.h*0.55} ${fl.w*0.8} ${fl.h*0.85} Q ${fl.w/2} ${fl.h*1.05} ${fl.w*0.2} ${fl.h*0.85} Q ${fl.w*0.05} ${fl.h*0.55} ${fl.w/2} 0 Z` }
                        }
                    }
                }
                Shape {
                    id: shield
                    anchors.fill: parent
                    // A clear pick glows in its own colour, pulsing.
                    property real pulse: 0.6
                    SequentialAnimation on pulse {
                        running: card.glow; loops: Animation.Infinite
                        NumberAnimation { from: 0.4; to: 1.0; duration: 900; easing.type: Easing.InOutSine }
                        NumberAnimation { from: 1.0; to: 0.4; duration: 900; easing.type: Easing.InOutSine }
                    }
                    layer.enabled: card.glow
                    layer.effect: MultiEffect {
                        shadowEnabled: true
                        shadowColor: card.t.edge
                        shadowBlur: 1.0
                        shadowOpacity: shield.pulse
                        shadowHorizontalOffset: 0; shadowVerticalOffset: 0
                        blurMax: 48
                        autoPaddingEnabled: true
                    }
                    preferredRendererType: Shape.CurveRenderer
                    ShapePath {
                        strokeColor: card.t.edge; strokeWidth: 1.5
                        fillGradient: LinearGradient {
                            x1: 0; y1: 0; x2: 0; y2: card.height
                            GradientStop { position: 0; color: card.t.hi }
                            GradientStop { position: 1; color: card.t.lo }
                        }
                        PathSvg { path: `M ${card.width/2} 0 L ${card.width} ${card.height*0.14} L ${card.width} ${card.height*0.5} Q ${card.width} ${card.height*0.84} ${card.width/2} ${card.height} Q 0 ${card.height*0.84} 0 ${card.height*0.5} L 0 ${card.height*0.14} Z` }
                    }
                    ShapePath {  // dark plate inside the rim
                        readonly property real m: card.width * 0.13
                        strokeColor: "transparent"
                        fillColor: "#ee15181c"
                        PathSvg { path: `M ${card.width/2} ${card.height*0.1} L ${card.width*0.87} ${card.height*0.21} L ${card.width*0.87} ${card.height*0.5} Q ${card.width*0.87} ${card.height*0.77} ${card.width/2} ${card.height*0.89} Q ${card.width*0.13} ${card.height*0.77} ${card.width*0.13} ${card.height*0.5} L ${card.width*0.13} ${card.height*0.21} Z` }
                    }
                }
                Text {
                    anchors.centerIn: parent
                    anchors.verticalCenterOffset: -card.height * 0.03
                    text: card.modelData.score ? Math.round(card.modelData.score) : "–"
                    color: "white"; font.pixelSize: card.width * 0.36; font.bold: true
                }
                Rectangle {  // pack rank, gold ring for v2's pick
                    visible: !!card.modelData.rank
                    width: card.width * 0.38; height: width; radius: width / 2
                    x: card.width * 0.78; y: -height * 0.15
                    color: "#15181c"
                    border.color: card.modelData.rank === 1 ? "#ffd54f" : "#80868e"
                    border.width: card.modelData.rank === 1 ? 2 : 1
                    Text {
                        anchors.centerIn: parent
                        text: card.modelData.rank
                        color: card.modelData.rank === 1 ? "#ffd54f" : "#d0d4d8"
                        font.pixelSize: parent.width * 0.55; font.bold: true
                    }
                }
            }
        }

        Repeater {  // stats panel over the top of the art
            model: root.data ? root.data.cards : []
            delegate: Rectangle {
                id: panel
                visible: !root.overCard
                required property var modelData
                required property int index
                readonly property int col: index % root.g.perRow
                readonly property int row: Math.floor(index / root.g.perRow)
                readonly property var t: win.medal(modelData)
                x: win.width * (root.g.x0 + col * root.g.dx) + win.unit * 0.3
                y: win.height * (root.g.y0 + row * root.g.dy + root.g.cardH * 0.1)
                width: stats.implicitWidth + win.unit * 0.8
                height: stats.implicitHeight + win.unit * 0.4
                radius: win.unit * 0.4
                color: "#cc0e1215"
                border.color: Qt.rgba(Qt.color(t.edge).r, Qt.color(t.edge).g, Qt.color(t.edge).b, 0.6)
                Column {
                    id: stats
                    anchors.centerIn: parent
                    Text {
                        text: (panel.modelData.gihwr ? "GIH " + Number(panel.modelData.gihwr).toFixed(1) : "GIH n/a") + (panel.modelData.upstream ? "  ·  U" + panel.modelData.upstream : "")
                        color: "#f0f0f0"; font.pixelSize: win.unit * 0.72; font.weight: Font.DemiBold
                    }
                    Text {
                        visible: !!panel.modelData.orders
                        text: (panel.modelData.orders || "").replace("Power ", "Pow ").replace(/#/g, "")
                        color: "#b8bec6"; font.pixelSize: win.unit * 0.65
                    }
                }
            }
        }
    }
}
