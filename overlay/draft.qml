// Badges over Arena's draft cards: a transparent, click-through layer-shell surface (niri overlay layer).
// Run with overlay/run.sh. OVERLAY_TEST=1 draws 14 dummy badges to check alignment without a draft.
// The API sorts the pack the way Arena displays it. Arena has two draft layouts; one screen pixel tells them apart.
// ponytail: assumes Arena fills the output. Grids measured on 2560x1440, 2026-11-02.
import QtQuick
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

        Text {
            visible: !!root.data && !!root.data.branch
            x: win.width * root.g.x0
            y: win.height * root.g.y0 - height - 6
            text: root.data ? root.data.branch : ""
            color: "white"; style: Text.Outline; styleColor: "black"
            font.pixelSize: win.height * 0.022; font.bold: true
        }

        Repeater {
            model: root.data ? root.data.cards : []
            delegate: Rectangle {
                required property var modelData
                required property int index
                readonly property int col: index % root.g.perRow
                readonly property int row: Math.floor(index / root.g.perRow)
                y: win.height * (root.g.y0 + row * root.g.dy + root.g.cardH * 0.1)
                x: win.width * (root.g.x0 + col * root.g.dx) + 4
                width: badge.implicitWidth + 10
                height: badge.implicitHeight + 4
                radius: 4
                color: modelData.rank === 1 ? "#d9c8a000" : modelData.rank <= 3 ? "#d92e7d32" : "#b3202020"
                border.color: modelData.rank === 1 ? "#ffe082" : "transparent"
                border.width: 2

                Column {
                    id: badge
                    anchors.centerIn: parent
                    spacing: 0
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: modelData.rank ? "#" + modelData.rank + "  " + Number(modelData.score).toFixed(1) : "–"
                        color: "white"; font.pixelSize: win.height * 0.017; font.bold: true
                    }
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: (modelData.orders || "").replace("Power", "Pow") + (modelData.upstream ? "  U" + modelData.upstream : "")
                        color: "#e0e0e0"; font.pixelSize: win.height * 0.0095
                    }
                    Text {
                        visible: root.showNames
                        width: win.width * root.g.cardW - 8
                        horizontalAlignment: Text.AlignHCenter
                        elide: Text.ElideRight
                        text: modelData.name
                        color: "#bdbdbd"; font.pixelSize: win.height * 0.012
                    }
                }
            }
        }
    }
}
