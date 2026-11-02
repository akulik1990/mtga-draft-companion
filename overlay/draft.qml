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
        // deck: the deck list's area [x0, y0, x1, y1]; hovering a card there opens Arena's preview over the pack too.
        side: { x0: 0.145, dx: 0.1025, cardW: 0.09, y0: 0.1707, dy: 0.2489, cardH: 0.227, perRow: 5, deck: [0.785, 0.15, 1.0, 0.92], panelBottom: 0.98 },
        bottom: { x0: 0.1715, dx: 0.0835, cardW: 0.069, y0: 0.1636, dy: 0.1858, cardH: 0.17, perRow: 8, deck: [0.0, 0.655, 1.0, 1.0], panelBottom: 0.55 }  // deck cards start under the Deck / Confirm Pick bar
    })
    // Arena's layout toggle in the top bar draws the current layout: deck on the side = bottom half filled, deck
    // below = right third filled. A sits in its lower left, B in its upper right (fractions of a 16:9 screen).
    readonly property var probeA: [0.830, 0.046]
    readonly property var probeB: [0.840, 0.027]
    function probeAt(p) { return Math.round(win.screen.x + p[0] * win.width) + "," + Math.round(win.screen.y + p[1] * win.height); }
    property string layoutName: "side"
    property string lastSeen: "side"
    readonly property var g: layouts[layoutName]
    property bool showNames: false // card name on each badge, to spot a wrong order at a glance

    property string screenName: Quickshell.env("OVERLAY_SCREEN") || "DP-1"
    property string api: (Quickshell.env("OVERLAY_API") || "http://localhost:17173") + "/api/overlay"
    property bool test: !!Quickshell.env("OVERLAY_TEST")  // 1 / gold / fire / bluefire: dummy pack, first card in that tier
    property var data: null
    property bool arenaFocused: false  // Arena is the active window on our output (the overlay shows)
    property bool pointerInArena: false  // Arena has keyboard focus; with niri's focus-follows-mouse, the pointer is over it
    property real px: -1  // pointer, in screen pixels (from overlay/pointer.py)
    property real py: -1
    // Arena opens a large preview over the pack while the pointer is on a card; the badges step aside.
    readonly property bool overCard: {
        // X11 keeps reporting the last position once the pointer leaves Arena; trust it only while Arena has focus.
        if (!data || !win.screen || test || !pointerInArena) return false;
        const fx = (px - win.screen.x) / win.width, fy = (py - win.screen.y) / win.height;
        const d = g.deck;
        if (fx >= d[0] && fx <= d[2] && fy >= d[1] && fy <= d[3]) return true;
        const col = Math.floor((fx - g.x0) / g.dx), row = Math.floor((fy - g.y0) / g.dy);
        return col >= 0 && col < g.perRow && row >= 0 && fx - g.x0 - col * g.dx <= g.cardW
            && fy - g.y0 - row * g.dy <= g.cardH && row * g.perRow + col < (data.cards || []).length;
        // ponytail: a basic land Arena sometimes shows after the last card is not in the log, so hovering it does not hide the overlay.
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
            cards.push({ name: "card " + (i + 1), rank: i + 1, score: 60 - i, orders: "Fit #" + (i + 1) + " · Power #" + (i + 1), upstream: i + 1,
                         fit: i + 1, power: i + 1, heat: i === 0 ? ({ fire: 1, bluefire: 2 })[Quickshell.env("OVERLAY_TEST")] || 0 : 0, gihwr: 60 - i, wheel: 10 * i });
        return { cards: cards, branch: "Stay: card 1 (not card 2)" };
    }

    Timer {
        interval: 1000; running: true; repeat: true; triggeredOnStart: true
        onTriggered: {
            if (root.test) { root.data = root.dummy(); root.arenaFocused = true; return; }
            root.poll();
            // A hover preview can cover the probe pixel; read the layout only while nothing is hovered.
            if (root.arenaFocused && !root.overCard && !pixel.running) pixel.running = true;
        }
    }

    Timer {  // window state, faster than the pack poll so hover hiding follows the pointer closely
        interval: 300; running: !root.test; repeat: true; triggeredOnStart: true
        onTriggered: if (!focus.running) focus.running = true
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
                    root.pointerInArena = root.arenaFocused && !!win.is_focused;
                } catch (e) { root.arenaFocused = false; }
            }
        }
    }

    Process {
        id: pixel
        command: ["sh", "-c", "for p in " + root.probeAt(root.probeA) + " " + root.probeAt(root.probeB) + "; do grim -t ppm -g \"$p 1x1\" - | tail -c 3 | od -An -tu1; done"]
        stdout: StdioCollector {
            onStreamFinished: {
                const v = text.trim().split(/\s+/).map(Number);
                if (v.length !== 6 || root.overCard)
                    return;
                const a = v[0] + v[1] + v[2], b = v[3] + v[4] + v[5];  // brightness: icon fill ~370-400, background ~90-180
                const seen = b > 300 && a < 250 ? "bottom" : a > 300 && b < 250 ? "side" : "";
                if (!seen)
                    return;  // icon covered or not on screen: keep the current layout
                if (seen === root.lastSeen)
                    root.layoutName = seen;  // two readings in a row before switching
                root.lastSeen = seen;
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
        function tierName(c) {
            if (c.rank === 1) return c.heat === 2 ? "bluefire" : c.heat === 1 ? "fire" : "gold";
            return ["", "", "silver", "bronze"][c.rank] || "grey";
        }
        function medal(c) { return medals[tierName(c)]; }
        readonly property int unit: Math.round(win.height * 0.017)

        readonly property var wubrg: ({ W: "#efe3b8", U: "#4a90d9", B: "#a58fcf", R: "#e0583a", G: "#3fa35c" })
        readonly property var pn: root.data && root.data.panel ? root.data.panel : null

        // Left panel: v2's pick and why, likeliest lanes, open colours, pool. Sits in the empty strip left of the pack.
        Rectangle {
            id: side
            visible: !!win.pn && !root.overCard  // Arena's preview can reach over it too
            x: win.width * 0.008
            y: win.height * 0.085  // just under Arena's top menu bar
            width: win.width * (root.g.x0 - 0.018)
            // Never over the deck list (with the deck below the pack it starts at mid-screen): the contents shrink to fit.
            readonly property real room: win.height * root.g.panelBottom - y - win.unit * 0.5
            readonly property real fit: Math.min(1, (room - win.unit * 1.2) / Math.max(1, col.implicitHeight))
            height: col.implicitHeight * fit + win.unit * 1.2
            radius: win.unit * 0.6
            color: "#d90e1215"
            border.color: "#30ffffff"

            Column {
                id: col
                x: win.unit * 0.6; y: win.unit * 0.6
                width: (side.width - win.unit * 1.2) / side.fit
                scale: side.fit; transformOrigin: Item.TopLeft
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
                        Line {  // win rate in each likely pair: ✕ cannot be cast there, – too few games
                            visible: (modelData.lanes || []).length > 0
                            leftPadding: win.unit * 1.5
                            textFormat: Text.StyledText
                            text: (modelData.lanes || []).map(l => "<b>" + l.lane + "</b> " + (l.wr === null ? "✕" : l.wr === "few" ? "–" : Number(l.wr).toFixed(1))).join("&nbsp;&nbsp;")
                            color: "#9aa3ad"; font.pixelSize: win.unit * 0.64
                        }
                    }
                }
                Line {  // how top drafters split between v2's top three when #1 leads by this much
                    visible: !!win.pn && !!win.pn.close && !!win.pn.close.took
                    text: win.pn && win.pn.close && win.pn.close.took
                          ? (win.pn.close.gap < 1 ? "Close call" : win.pn.close.gap < 4 ? "Lead" : "Clear") + " +" + win.pn.close.gap
                            + " · top drafters take #1 " + Math.round(win.pn.close.took[0])
                            + "% #2 " + Math.round(win.pn.close.took[1]) + "% #3 " + Math.round(win.pn.close.took[2]) + "%"
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
                Row {  // open colours (signal from the passing neighbour) beside your likeliest lanes
                    spacing: win.unit * 0.6
                    Column {
                        width: (col.width - win.unit * 0.6) * 0.48
                        spacing: 2
                        Heading { text: "OPEN" }
                        Row {
                            id: sig
                            readonly property var vals: win.pn ? win.pn.signals : ({})
                            readonly property real peak: Math.max(0.01, ...Object.values(vals).map(Math.abs))
                            readonly property bool any: Object.values(vals).some(v => v !== 0)
                            spacing: win.unit * 0.15
                            Repeater {
                                model: ["W", "U", "B", "R", "G"]
                                Column {
                                    required property string modelData
                                    readonly property real v: sig.vals[modelData] || 0
                                    Item {  // up = open, down = cut
                                        width: (col.width * 0.48 - win.unit * 1.2) / 5; height: win.unit * 1.6
                                        Rectangle { y: parent.height / 2; width: parent.width; height: 1; color: "#40ffffff" }
                                        Rectangle {
                                            width: parent.width * 0.7; x: parent.width * 0.15
                                            height: Math.abs(parent.parent.v) / sig.peak * parent.height / 2
                                            y: parent.parent.v >= 0 ? parent.height / 2 - height : parent.height / 2
                                            radius: 1; color: win.wubrg[modelData]; opacity: parent.parent.v >= 0 ? 1 : 0.45
                                        }
                                    }
                                    Text { anchors.horizontalCenter: parent.horizontalCenter; text: modelData; color: win.wubrg[modelData]; font.pixelSize: win.unit * 0.55; font.bold: true }
                                }
                            }
                        }
                    }
                    Column {
                        width: (col.width - win.unit * 0.6) * 0.52
                        spacing: 1
                        Heading { text: "LANES" }
                        Repeater {
                            model: win.pn ? win.pn.lanes : []
                            Row {
                                required property var modelData
                                spacing: win.unit * 0.25
                                Text { text: modelData.lane; width: win.unit * 1.2; color: "#e8eaed"; font.pixelSize: win.unit * 0.6; font.bold: true }
                                Item {
                                    width: (col.width - win.unit * 0.6) * 0.52 - win.unit * 3.2; height: win.unit * 0.4
                                    anchors.verticalCenter: parent.verticalCenter
                                    Rectangle { anchors.fill: parent; radius: height / 2; color: "#2a2f35" }
                                    Row {
                                        height: parent.height
                                        Rectangle { width: parent.parent.width * modelData.p / win.pn.lanes[0].p / 2; height: parent.height; color: win.wubrg[modelData.lane[0]] }
                                        Rectangle { width: parent.parent.width * modelData.p / win.pn.lanes[0].p / 2; height: parent.height; color: win.wubrg[modelData.lane[1]] }
                                    }
                                }
                                Text { text: Math.round(modelData.p * 100) + "%"; width: win.unit * 1.5; horizontalAlignment: Text.AlignRight; color: "#c8cdd3"; font.pixelSize: win.unit * 0.55 }
                            }
                        }
                    }
                }

                // Your deck in the likeliest pair against top players' typical deck there (tools/deck_targets.py).
                Item { width: 1; height: win.unit * 0.3; visible: !!deck.d }
                Heading { id: deck; readonly property var d: win.pn ? win.pn.deck : null; visible: !!d; text: d ? "YOUR " + d.lane + " DECK vs TOP PLAYERS" : "" }
                Repeater {
                    model: deck.d ? [["creatures", "Creatures"], ["removal", "Removal"], ["card_advantage", "Card draw"], ["evasion", "Evasion"], ["two_drops", "2-drops"], ["five_plus", "5+ drops"]] : []
                    Row {
                        required property var modelData
                        readonly property real have: deck.d.have[modelData[0]]
                        readonly property real want: deck.d.target[modelData[0]]
                        readonly property real due: want * deck.d.pace  // roughly where a drafter is by now
                        spacing: win.unit * 0.3
                        Text { text: modelData[1]; width: win.unit * 3.6; color: "#c8cdd3"; font.pixelSize: win.unit * 0.6 }
                        Item {
                            width: col.width - win.unit * 7.2; height: win.unit * 0.45
                            anchors.verticalCenter: parent.verticalCenter
                            Rectangle { anchors.fill: parent; radius: height / 2; color: "#2a2f35" }
                            Rectangle {
                                width: parent.width * Math.min(1, have / Math.max(1, want)); height: parent.height; radius: height / 2
                                color: have >= due ? "#5fae6e" : have >= due * 0.6 ? "#d6a84a" : "#d0613f"
                            }
                        }
                        Text { text: have + " / " + Math.round(want); width: win.unit * 2.6; horizontalAlignment: Text.AlignRight; color: "#e8eaed"; font.pixelSize: win.unit * 0.6 }
                    }
                }

                // Curve of your spells in that pair; under each mana value, top players' typical count.
                Item { width: 1; height: win.unit * 0.5; visible: !!deck.d }
                Row {
                    id: curve
                    visible: !!deck.d
                    readonly property var slots: ["1", "2", "3", "4", "5", "6+"]
                    readonly property var c: deck.d ? deck.d.curve : ({})
                    readonly property var t: deck.d ? deck.d.target_curve : ({})
                    readonly property real peak: Math.max(1, ...slots.map(k => c[k] || 0))
                    spacing: win.unit * 0.25
                    Repeater {
                        model: curve.slots
                        Column {
                            required property string modelData
                            readonly property real n: curve.c[modelData] || 0
                            readonly property real want: curve.t[modelData] || 0
                            spacing: 1
                            Item {
                                width: (col.width - win.unit * 1.3) / 6; height: win.unit * 1.9
                                Rectangle {  // yours, coloured against top players' count at this point of the draft
                                    readonly property real due: want * (deck.d ? deck.d.pace : 0)
                                    anchors.bottom: parent.bottom; anchors.horizontalCenter: parent.horizontalCenter
                                    width: parent.width * 0.7; radius: 2
                                    height: Math.max(2, n / curve.peak * parent.height)
                                    color: n > want + 1 ? "#6f8aa6" : n >= due - 0.5 ? "#5fae6e" : n >= due * 0.6 ? "#d6a84a" : "#d0613f"
                                }
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    y: parent.height - Math.max(2, n / curve.peak * parent.height) - height
                                    text: n; color: "#e8eaed"; font.pixelSize: win.unit * 0.55; font.bold: true
                                }
                            }
                            Text {  // mana value, then top players' typical count
                                anchors.horizontalCenter: parent.horizontalCenter
                                textFormat: Text.StyledText
                                text: modelData + " <font color='#6b737c'>/" + Math.round(want) + "</font>"
                                color: "#8f98a3"; font.pixelSize: win.unit * 0.55
                            }
                        }
                    }
                }
            }
        }

        Repeater {
            model: root.data ? root.data.cards : []
            delegate: Shield {
                required property var modelData
                required property int index
                readonly property int col: index % root.g.perRow
                readonly property int row: Math.floor(index / root.g.perRow)
                visible: !root.overCard
                tier: win.tierName(modelData)
                glow: win.clear(modelData)
                score: modelData.score
                rank: modelData.rank || 0
                // On the card's bottom edge, clear of art and rules text.
                width: win.width * root.g.cardW * 0.34
                height: width * 1.15
                x: win.width * (root.g.x0 + col * root.g.dx + root.g.cardW / 2) - width / 2
                y: win.height * (root.g.y0 + row * root.g.dy + root.g.cardH) - height * 0.45
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
                readonly property real k: Math.min(1, root.g.cardW / 0.09)  // smaller text on the narrower cards of the deck-below layout
                readonly property real maxW: win.width * root.g.cardW - win.unit * 1.2
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
                        width: Math.min(implicitWidth, panel.maxW); elide: Text.ElideRight
                        text: (panel.modelData.gihwr ? "GIH " + Number(panel.modelData.gihwr).toFixed(1) : "GIH n/a") + (panel.modelData.upstream ? " · U" + panel.modelData.upstream : "")
                        color: "#f0f0f0"; font.pixelSize: win.unit * 0.72 * panel.k; font.weight: Font.DemiBold
                    }
                    Text {
                        width: Math.min(implicitWidth, panel.maxW); elide: Text.ElideRight
                        text: (panel.modelData.orders || "").replace("Power ", "Pow ").replace(/#/g, "")
                        color: "#b8bec6"; font.pixelSize: win.unit * 0.65 * panel.k
                    }
                    Text {  // chance the card comes back on the wheel
                        visible: panel.modelData.wheel != null
                        text: "↻ " + Math.round(panel.modelData.wheel || 0) + "%"
                        color: (panel.modelData.wheel || 0) >= 50 ? "#9fd3a8" : "#b8bec6"; font.pixelSize: win.unit * 0.65 * panel.k
                    }
                }
            }
        }
    }
}
