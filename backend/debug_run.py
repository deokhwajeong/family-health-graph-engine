from app.graph_builder import build_example_graph


def main():
    print("Building example graph...")
    G = build_example_graph()

    print(f"Total nodes: {len(G.nodes())}")
    print(f"Total edges: {len(G.edges())}")

    print("\nNodes:")
    for node, attrs in G.nodes(data=True):
        print(f"  {node}  {attrs}")

    print("\nEdges:")
    for u, v, attrs in G.edges(data=True):
        print(f"  {u} -> {v}  {attrs}")


if __name__ == "__main__":
    main()
