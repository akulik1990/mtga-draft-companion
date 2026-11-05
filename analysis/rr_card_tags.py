"""Hand tags (read from each card's text, 2026-11-05) for the cards in 100 winning and 100 losing UB Premier decks that run Recursive Recruitment.
Tags: mill = puts own cards in the graveyard directly (mill, self-mill ETB); surveil = card selection that can bin cards (surveil/scry);
gy_payoff = gets better with a full graveyard (threshold, counts cards, flashback scaling); gy_recursion = brings specific cards back;
jace_enabler = empowers a Jace token; jace_payoff = rewards controlling a Jace / planeswalker; removal = kills or exiles a creature;
tempo = bounce, tap/stun, counter, -X/-0; draw = card advantage; ramp = landcycler, mana rock, cost reducer; dual = UB dual land;
defence = a body meant to block (defender, big toughness, lifegain ETB).
DOUBT marks a call Akshin should review."""

TAGS = {
    "Recursive Recruitment": "gy_payoff",  # the payoff under test; its flashback is 8 mana
    "Extended Absence": "removal",
    "Mindseeker Oculus": "jace_enabler",
    "Theorix Charm": "removal tempo mill",  # modal: -2/-2, soft counter, or mill 3 + draw. DOUBT: one card, three roles
    "Surveillance Phantasm": "surveil defence",  # 0/3? defender flyer, surveil for {3}{U}
    "Apex Witchstalker": "ramp defence",  # landcycler; 6-drop with lifegain
    "Icy Reception": "tempo",
    "Last Gasp": "removal",
    "Void Extrapolator": "mill gy_payoff",  # mills 3 via its prepared half; threshold 3/3
    "Theoretical Necromancer": "gy_recursion",
    "Unsummon": "tempo",
    "Theorix Metamage": "mill gy_payoff",  # mill 3; threshold flying
    "Undulating Witness": "ramp",
    "Twinned Vision": "draw",  # flashback draw; DOUBT: also a discard outlet for the graveyard
    "Protege's Awakening": "jace_enabler draw",
    "Paradox Shaper": "mill",  # mills 3 every upkeep it re-prepares: the strongest repeat enabler
    "Infinite Coursework": "removal",  # aura that locks a creature
    "Theorix Annex": "dual",
    "Rank Rat": "",  # discard ETB for the opponent; neither package
    "Silence the Echo": "removal",
    "Break Under Pressure": "removal",
    "Countersculpt": "tempo jace_enabler jace_payoff",  # counterspell; behold a Jace
    "Room of Refuge": "dual",
    "Fblthp, Impossibly Lost": "draw",
    "Plan for All Outcomes": "tempo jace_enabler",
    "Fatehold Chronologist": "surveil",
    "Gallia, Tragic Host": "gy_recursion",  # returns herself by exiling a creature card from the graveyard
    "Proft, Sinister Mastermind": "gy_payoff",  # can only be cast at threshold
    "Multiply by Zero": "removal",  # 0/0 until end of turn: kills unless pumped
    "Rewrite Regrets": "gy_recursion jace_enabler",
    "Way of the Mind Sculptor": "jace_enabler jace_payoff draw",
    "Arni, Humble Scribe": "draw",  # looter: also bins cards. DOUBT: mill-ish
    "Null Summoner": "gy_payoff",  # threshold lets you cast the stolen card
    "Proft, Consulting Detective": "draw",  # pays off surveil
    "Keeper of the Quiet Hour": "jace_enabler",
    "Geist of Saint Thalia": "ramp",  # noncreature spells cost 1 less (Recruitment flashback 7)
    "Murmuring Volume": "ramp draw",  # mana rock + rummage
    "Traxos, Academy Guardian": "",  # cheap flyer after a noncreature spell; DOUBT: self cost reducer only
    "Semester Foreseer": "surveil",
    "Sphinx's Approach": "draw",
    "Hapatra, the Desert Fang": "removal gy_payoff",  # -1/-1 counters = greatest MV in your graveyard (BG splash)
    "Ruric Thar, Biomagus": "",  # 6-drop flyer
    "Shipwreck Marsh": "dual",
    "Tam's Resistance": "jace_enabler",
    "Mabel, Bitter Recluse": "defence",  # 1-mana deathtouch
    "Tinybones, Pocket Nuisance": "",
    "Liliana the Repentant": "mill gy_recursion",  # mills 2 per creature/planeswalker entering; exhaust reanimation
    "Theorist's Sanctum": "jace_enabler",  # land
    "Cryotheory Adept": "tempo",  # stun from the graveyard (uses the graveyard, does not need it full)
    "Tetsuko Umezawa, Fugitive": "",
    "Winter, Tormented Loner": "removal gy_payoff",  # edict; +1/+0 per creature/planeswalker card in graveyard
    "Screeching Soulbreaker": "",
    "Dark Matter Manipulator": "mill gy_payoff",  # mill 3 ETB; +2/+0 per seven cards
    "Blessed Ghoul": "gy_recursion defence",
    "Lyra, Tolarian Archangel": "draw",  # makes angels on 3+ draws; draws on damage
    "Uldaros Theorix": "gy_payoff",  # 7-mana: copies graveyard cards. DOUBT: expensive payoff like Recruitment
    "Rise of the Deathbringer": "removal draw",  # sweeper or draw
    "Jace, Reality Sculptor": "jace_payoff jace_enabler",
    "Archive Arbiter": "",  # 6-mana flyer; noncreature removal or life
    "Yuriko, Hope from the Shadows": "tempo surveil",  # -X/-0 by graveyard size, or surveil 2. DOUBT: also a gy payoff
    "Prudent Fateseer": "surveil",
    "Diviner of Victory": "tempo surveil",
    "Solve for Disappointment": "jace_enabler",
    "Sphinx of False Conclusions": "draw",
    "Vraska's Final Mercy": "removal jace_enabler",
    "Way of the Cryomancer": "jace_enabler",
    "Mind Meanderer": "removal jace_payoff",  # fights on ETB; UG
    "Lich's Relic": "removal",
    "Sanctum Lurker": "jace_enabler jace_payoff",
    "Way of the Deathbringer": "jace_enabler",
    "Gideon the Oathless": "",
    "Theorist's Proxy": "jace_enabler",
    "Jace's Machinations": "jace_enabler jace_payoff",
    "Variable Chaser": "draw",
    "Hapatra, the Desert Frost": "tempo",
    "Divining Duelist": "tempo",
    "Cast Away Doubt": "draw",
    "Garruk, Veiled Butcher": "removal",
    "Seasoned Cryomancer": "tempo draw",
    "The Theorist, Jace Beleren": "draw tempo",
    "Overwrite the Multiverse": "removal jace_enabler",
    "The Echoverse Fulcrum": "removal",
    "Primal Witchstalker": "mill ramp",
    "Something Worth Saving": "mill",
    "Eye of Jace": "surveil gy_payoff",
    "Consider": "surveil draw",
    "Chandra, Chill of Compliance": "surveil ramp tempo",  # +1 surveil/regrowth, +1 mana, -X stun
    "Sublime Epiphany": "tempo draw",
    "Perfected Theory": "tempo",
    "Loot, the Anomaly": "gy_payoff",
    "Fatehold Charm": "tempo jace_enabler",
    "Way of the Necromancer": "jace_enabler",
    "Hexhaven Battalion": "ramp jace_enabler",
    "Terminal Criticism": "removal",
    "Vindictive Triumph": "removal",
}
TAGS = {k: set(v.split()) for k, v in TAGS.items()}
