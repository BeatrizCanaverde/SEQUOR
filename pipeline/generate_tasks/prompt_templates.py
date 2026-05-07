agenda_template = '''Create a possible agenda for a day in the life of the following persona:

{persona}

Note:
1. Identify several activities or tasks that the persona might engage in throughout their day. Provide a detailed description of each activity or task.
2. The agenda should be specific and tailored to the persona's characteristics, interests, and lifestyle.
3. Your output should start with "Agenda: " and list the activities in chronological order. Identify each activity in a new line with the markdown divider "###"'''




activities_template = '''Next, you are given the description of a persona and an activity/task from their daily agenda. Elaborate on how this activity/task might unfold, setting the stage for interesting questions the persona could naturally wonder about or ask an assistant for clarification.
Your ultimate goal is to generate a sequence of open-ended, creative questions that build upon the scenario. Put yourself in the position of the persona; each question should feel as if the persona is asking it in real time to someone or an AI assistant. Importantly, no question should depend on or assume answers to previous ones.

Persona: {persona}
Activity/Task: {activity}

Note:
1. The questions can include details such as the location where the actions take place, people involved, time of day, emotions, challenges, or other relevant aspects that make the scenario vivid and engaging.
2. Ensure the questions are coherent and consistent with both the persona and the activity/task. Avoid contradictions.
3. Write each question on a new line, preceded by the markdown divider "###"'''

