// Badges over Arena's draft cards: a transparent, click-through layer-shell surface (niri overlay layer).
// Run with overlay/run.sh. OVERLAY_TEST=1 draws 14 dummy badges to check alignment without a draft.
// The API sorts the pack the way Arena displays it. Arena has two draft layouts; one screen pixel tells them apart.
// ponytail: assumes Arena fills the output. Grids measured on 2560x1440, 2026-11-02.
import QtQuick
import QtQuick.Shapes
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

        // Draftsmith-style tiers on v2's score, in set SDs from the set's mean GIH WR. v2's scores run about one SD
        // under raw win rates early (they average over lanes a card cannot be cast in), hence the low cut-offs.
        readonly property var tiers: [
            { min: 1, hi: "#ffd27a", lo: "#e0441c", edge: "#ffb347" },    // fire
            { min: 0, hi: "#fff1a8", lo: "#c9971c", edge: "#ffd54f" },    // gold
            { min: -1, hi: "#ffffff", lo: "#9aa4ae", edge: "#d5dbe1" },   // silver
            { min: -2, hi: "#ffc58f", lo: "#a5612c", edge: "#e0955a" },   // bronze
            { min: -99, hi: "#6a7078", lo: "#2e3238", edge: "#80868e" }   // weak
        ]
        function tier(score) {
            if (!score || !root.data || !root.data.std) return tiers[4];
            const z = (score - root.data.mean) / root.data.std;
            return tiers.find(t => z >= t.min);
        }
        readonly property int unit: Math.round(win.height * 0.017)

        Rectangle {
            visible: !!root.data && !!root.data.branch
            x: win.width * root.g.x0
            y: win.height * root.g.y0 - height - win.unit * 0.6
            width: branchText.implicitWidth + win.unit * 1.4
            height: branchText.implicitHeight + win.unit * 0.6
            radius: height / 2
            color: "#d90e1215"
            border.color: "#40ffffff"
            Text {
                id: branchText
                anchors.centerIn: parent
                text: root.data ? root.data.branch : ""
                color: "#f0f0f0"; font.pixelSize: win.unit * 1.05; font.weight: Font.DemiBold
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
                readonly property var t: win.tier(modelData.score)
                // Shield on the card's bottom edge, clear of art and rules text.
                width: win.width * root.g.cardW * 0.34
                height: width * 1.15
                x: win.width * (root.g.x0 + col * root.g.dx + root.g.cardW / 2) - width / 2
                y: win.height * (root.g.y0 + row * root.g.dy + root.g.cardH) - height * 0.45

                Shape {
                    id: shield
                    anchors.fill: parent
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
    }
}
