#!/usr/bin/env python3
"""MKAngel Interactive Chat — terminal interface."""

import sys
sys.path.insert(0, ".")

from app.conductor import AngelConductor


def main():
    print()
    print("\033[36m" + "=" * 55 + "\033[0m")
    print("\033[1m  MKAngel — Grammar Language Model\033[0m")
    print("\033[36m" + "=" * 55 + "\033[0m")
    print()
    print("  Booting...", end=" ", flush=True)

    conductor = AngelConductor().awaken()
    info = conductor.angel.introspect() if conductor.angel else {}
    print("done.")
    print("  %d domains | %d grammars | %d words" % (
        len(info.get("domains_loaded", [])),
        info.get("total_grammars", 0),
        info.get("lexicon_size", 0),
    ))
    print()
    print("  Type a message. /help for commands. /exit to quit.")
    print()

    while True:
        try:
            user_input = input("\033[32myou>\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            break

        if user_input.startswith("/"):
            # Try conductor commands first
            result = conductor.handle_command(user_input)
            if result is not None:
                print("\033[36m%s\033[0m" % result)
                print()
                continue

            # Special commands
            cmd = user_input.split()[0].lower()

            if cmd == "/help":
                print("\033[36m  Commands:")
                print("    /parse <words>     Parse and show syntax tree")
                print("    /predict <words>   Predict what comes next")
                print("    /fugue <words>     Cross-domain fugue")
                print("    /invoke <angel>    Direct line to named angel")
                print("    /swarm <task>      Run swarm orchestration")
                print("    /status            Show Angel status")
                print("    /consent           Consent status")
                print("    /growth            Growth summary")
                print("    /language          Language setting")
                print("    /exit              Quit\033[0m")
                print()
                continue

            if cmd == "/parse":
                words = user_input.split()[1:]
                if words:
                    result = conductor.angel.parse(words)
                    print("\033[36m  Tags: %s" % result["tags"])
                    if result["tree"]:
                        print("  Tree: %s" % result["tree"])
                    print("\033[0m")
                else:
                    print("  Usage: /parse the cat sat")
                print()
                continue

            if cmd == "/predict":
                words = user_input.split()[1:]
                if words:
                    preds = conductor.angel.predict(words, domain="linguistic", horizon=5)
                    print("\033[36m  %d predictions:" % len(preds))
                    for p in preds[:5]:
                        print("    -> %s (%.2f)" % (p.get("predicted"), p.get("confidence", 0)))
                    print("\033[0m")
                else:
                    print("  Usage: /predict the cat")
                print()
                continue

            if cmd == "/fugue":
                words = user_input.split()[1:]
                if words:
                    fugue = conductor.angel.compose_fugue(words)
                    voices = fugue.get("voices", {})
                    if isinstance(voices, dict):
                        print("\033[36m  %d voices:" % len(voices))
                        for domain, derivations in voices.items():
                            print("    [%s] %d derivations" % (domain, len(derivations)))
                            for d in derivations[:2]:
                                if isinstance(d, dict):
                                    print("      %s (via %s)" % (d.get("output", "?"), d.get("rule", "?")))
                    print("\033[0m")
                else:
                    print("  Usage: /fugue energy transforms")
                print()
                continue

            if cmd == "/invoke":
                parts = user_input.split(None, 2)
                if len(parts) >= 3:
                    angel_name = parts[1]
                    message = parts[2]
                    result = conductor.invoke_angel(angel_name, message)
                    print("\033[36m  [%s] %s" % (
                        result.get("title", "?"),
                        result.get("response", "No response")[:500],
                    ))
                    print("\033[0m")
                else:
                    print("  Usage: /invoke gabriel Tell me about grammar")
                print()
                continue

            if cmd == "/swarm":
                task = user_input[len("/swarm "):].strip()
                if task:
                    print("  Running swarm (1 cycle)...", flush=True)
                    report = conductor.run_swarm(task, cycles=1)
                    if report:
                        cr = report.cycle_results[0]
                        print("\033[36m  Confidence: %.2f" % cr.results.confidence)
                        print("  Best agents: %s" % ", ".join(cr.best_agents))
                        for opt in report.options[:3]:
                            print("    %s" % opt)
                        print("\033[0m")
                else:
                    print("  Usage: /swarm How do bonds relate to syntax?")
                print()
                continue

            if cmd == "/status":
                status = conductor.get_status()
                active = [k for k, v in status["subsystems"].items() if v == "active"]
                print("\033[36m  %d/%d subsystems active" % (len(active), len(status["subsystems"])))
                print("  Active: %s" % ", ".join(active))
                print("\033[0m")
                print()
                continue

            print("  Unknown command. Type /help")
            print()
            continue

        # Normal message
        response = conductor.process(user_input)
        print("\033[35mangel>\033[0m %s" % response)
        print()

    # Shutdown
    msg = conductor.shutdown()
    print("\033[2m%s\033[0m" % msg)
    print()


if __name__ == "__main__":
    main()
