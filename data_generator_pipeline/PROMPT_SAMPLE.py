

prompt_template = '''
You are an expert AI assistant specializing in creating high-quality datasets for instruction fine-tuning Large Language Models. 
Your task is to transform the provided "Input Text" into a series of structured JSON objects, each containing an instruction, optional context, and a ground-truth output.

Return the result as a single JSON array, where each object follows the specified format.

**JSON Output Format:**
[
  {{
    "instruction": "<A clear command or question for the model, in Bahasa Indonesia.>",
    "input": "<A direct quote of 1-5 sentences from the 'Input Text' that contains the necessary information to answer the instruction. This can be an empty string if no specific context is needed.>",
    "output": "<A comprehensive, detailed, and well-explained response that fully answers the instruction. It should be generated *only* from the information in the 'input' field (or general knowledge if 'input' is empty). The language of the output must match the language of the instruction.>"
  }}
]

**Mandatory Rules You Must Follow:**
1. **Direct Quotation for Input**: The input field MUST be a direct quote extracted from the "Input Text". DO NOT PARAPHRASE OR REWRITE the text for the input field. It must be a verbatim copy of 1-5 relevant sentences.
2. **Language Consistency**: The instruction and output can be in Bahasa Indonesia.
3. **Comprehensive and Grounded Output**: The output must be a comprehensive and detailed answer that fully addresses the instruction. It must be generated exclusively from the information provided in the input field. If the input is empty, the output should still be a thorough, self-contained response.
4. **Detailed Responses**: Avoid overly simple or short answers. The output should be explanatory and provide as much detail as is supported by the context in the input field.
5. **Vary the Data**: Generate a mix of examples. Some should have a populated input field for context-dependent questions, and others should have an empty input ("") for more general instructions.
6. **JSON Only**: Your entire output must be a single, valid JSON array of objects. Do not include any explanatory text, acknowledgements, or any characters before the opening [ or after the closing ].
7. **Empty Source Text**: If the "Input Text" is empty or contains no meaningful information, you must return an empty JSON array [].

Input Text:

{text}
'''
