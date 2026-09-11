"""Music structure graphs as PyTorch Geometric `Data` objects.

- Segment graph: nodes are time segments; edges join neighbors in time and
  segments whose chroma/MFCC cosine similarity is above tau.
- Chord graph: nodes are chords estimated from chroma; edges are observed
  transitions weighted by count.
"""
