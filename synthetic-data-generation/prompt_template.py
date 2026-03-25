sub_concept = """
   TASK 
   You are a clinical psychologist. Given the symptom keyword extracted from DSM-5 and its examples, generate a specific description of the symptom keyword. 

   INSTRUCTION 
   1. Consider the clinical definition of the symptom keyword regarding depressive disorders.
   2. Generate a detailed description of that symptom. 
   - Written in clinical, direct words.
   - Refer to the examples provided. 
   - Generate at least ten descriptions.
   3. Output the examples in JSON format

   NOTE
   This task aims to connect formal DSM-5 criteria with actual symptom manifestations. Therefore, while some content might be sensitive, please generate descriptions using direct and realistic language representing how people genuinely experience these symptoms.

   OUTPUT FORMAT
   {{
   "Symptom": "Anger_Irritability",
   "Descriptions": ["Exhibiting increased irritability", "Feeling irritable or having fits of anger", "..."] 
   }}

   TARGET SYMPTOM
   {symptom}
   """

normalize = """
   You are a mental health professional.

   Given a raw social media post, identify the parts that describe emotional distress, psychiatric symptoms, or psychological struggles.  
   Rewrite them into a short, natural paragraph in the first-person perspective.

   INSTRUCTIONS:
   1. Focus only on parts describing emotional states, psychiatric symptoms, psychological struggles, or mental health distress.
      - Relevant symptoms may include: sadness, hopelessness, anxiety, fear, anger, guilt, worthlessness, low self-esteem, insomnia, fatigue, suicidal thoughts, loss of interest, irritability, or trauma-related distress.
   2. Ignore unrelated background stories, long situational explanations, or non-symptom content.
   3. If multiple symptom-relevant parts exist, summarize them naturally into a single, coherent paragraph (ideally 2–4 sentences).
   4. Maintain the original wording if it already expresses the symptoms clearly and concisely.
   5. Keep the output concise: ideally between 20 and 80 words. A slightly longer output (up to 100 words) is acceptable for complex posts.
   6. Do not add clinical interpretations unless they are explicitly mentioned by the user.
   7. If no symptom-relevant content is found, output: "No symptom-relevant content found."
   8. If the input is long, focus on identifying psychiatric symptoms that would be relevant for mental health diagnosis (e.g., emotional distress, cognitive or behavioral symptoms, functional impairment).  Summarize only these symptom-related parts into a short paragraph. Ignore unrelated background information, even if it appears contextually connected to the story.

   OUTPUT FORMAT:
   Return only the extracted or summarized paragraph, without any additional commentary or explanation.

   EXAMPLES:

   Input:  
   "So yesterday I went to the mall and bought a new pair of shoes. It was fun, but also I've been feeling really anxious around crowds lately, even when shopping."  
   Output:  
   "I feel really anxious around crowds, even during everyday activities like shopping."

   ---

   Input:  
   "I had a terrible day. I tried to go out and see my friends but I felt so anxious that I froze at the door. I just stood there, feeling like everyone would judge me if I stepped outside. It's been like this for weeks now. I end up staying home, feeling even more miserable and isolated. I don't know how to break this cycle."  
   Output:  
   "I've been feeling extremely anxious and unable to leave my house, overwhelmed by fear of being judged. This has left me feeling isolated and miserable for weeks."

   ---

   Input:  
   "I've been feeling worthless lately. Nothing I do seems good enough, and I don't think I'll ever make my parents proud."  
   Output:  
   "I've been feeling worthless lately. Nothing I do seems good enough, and I don't think I'll ever make my parents proud."

   ---

   Input:  
   "I'm exhausted and numb. I don't even care anymore. It’s like nothing matters and I can’t feel anything."  
   Output:  
   "I feel emotionally numb and exhausted. Nothing seems to matter anymore, and I can't bring myself to care about anything."

   ---

   Input:
   "I love pizza and video games. Can't wait for the new release next week!"

   Output:
   "No symptom-relevant content found."
   """

