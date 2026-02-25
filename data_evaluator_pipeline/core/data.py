# core/data.py — now backed by the shared BaseDataPoint
from shared.models import BaseDataPoint

# Alias so all existing evaluator code keeps working unchanged.
# BaseDataPoint already has: instruction, input, output, metadata, full_instruction
DataPoint = BaseDataPoint
