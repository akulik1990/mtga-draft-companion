// Badges over Arena's draft cards: a transparent, click-through layer-shell surface (niri overlay layer).
// Run with overlay/run.sh. OVERLAY_TEST=1 draws 14 dummy badges to check alignment without a draft.
// ponytail: assumes Arena fills the output and lays the pack out in log order, five per row. Calibrate below.
import QtQuick
import Quickshell
import Quickshell.Io
import Quickshell.Wayland

ShellRoot {
    id: root

    // Card grid as fractions of the screen, from a 16:9 capture of the draft screen.
    property real x0: 0.1436      // left edge of column 1
    property real dx: 0.1033      // column step
    property real cardW: 0.0907
    property real y0: 0.1682      // top edge of row 1
    property real dy: 0.2511      // row step
    property real cardH: 0.235
    property int perRow: 5
    property bool showNames: true // card name on each badge, to spot a wrong order at a glance

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
        }
    }

    Process {
        id: focus
        command: ["niri", "msg", "--json", "focused-window"]
        stdout: StdioCollector {
            onStreamFinished: {
                try { root.arenaFocused = (JSON.parse(text) || {}).app_id === "steam_app_2141910"; }
                catch (e) { root.arenaFocused = false; }
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
            x: win.width * root.x0
            y: win.height * root.y0 - height - 6
            text: root.data ? root.data.branch : ""
            color: "white"; style: Text.Outline; styleColor: "black"
            font.pixelSize: win.height * 0.022; font.bold: true
        }

        Repeater {
            model: root.data ? root.data.cards : []
            delegate: Rectangle {
                required property var modelData
                required property int index
                readonly property int col: index % root.perRow
                readonly property int row: Math.floor(index / root.perRow)
                x: win.width * (root.x0 + col * root.dx)
                y: win.height * (root.y0 + row * root.dy + root.cardH * 0.36)
                width: win.width * root.cardW
                height: badge.implicitHeight + 8
                radius: 4
                color: modelData.rank === 1 ? "#e6c8a000" : modelData.rank <= 3 ? "#e62e7d32" : "#cc202020"
                border.color: modelData.rank === 1 ? "#ffe082" : "transparent"
                border.width: 2

                Column {
                    id: badge
                    anchors.centerIn: parent
                    spacing: 0
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: modelData.rank ? "#" + modelData.rank + "  " + Number(modelData.score).toFixed(1) : "–"
                        color: "white"; font.pixelSize: win.height * 0.024; font.bold: true
                    }
                    Text {
                        anchors.horizontalCenter: parent.horizontalCenter
                        text: (modelData.orders || "").replace("Power", "Pow") + (modelData.upstream ? "  U" + modelData.upstream : "")
                        color: "#e0e0e0"; font.pixelSize: win.height * 0.013
                    }
                    Text {
                        visible: root.showNames
                        width: win.width * root.cardW - 8
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
