from .util import *

data = "MATCH (c:crop {name: 'Carissa macrocarpa'})-[:HAS]->(ecology)MATCH (ecology)-[:GROWS_IN]->(optimal)MATCH (optimal)-[:CONSIST_OF]->(temperature:temperature_required_optimal)RETURN temperature.min, temperature.max;"

tokens = load_vocab()