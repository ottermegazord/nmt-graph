

''' Query from Python '''

from py2neo import Graph
import re

graph = Graph(host="localhost", password="farmers@heart")
# a = graph.run("MATCH (c:crop) RETURN c.id, c.name LIMIT 4").data()
# print(a)

class CypherQuestion():

    def __init__(self, english_question_template, cypher_question_template, cypher_answer_template, ENGLISH_PATH, CYPHER_PATH):
        self.english_question = english_question_template
        self.cypher_question = cypher_question_template
        self.cypher_answer = cypher_answer_template
        self.var_list = graph.run(cypher_question_template).data()
        self.ENGLISH_PATH = ENGLISH_PATH
        self.CYPHER_PATH = CYPHER_PATH

    def generate_english_question(self):

        f = open(ENGLISH_PATH, "w")
        for i in self.var_list:
            output = self.english_question.format(**i) + '\n'
            print(output)
            f.write(output)

    def generate_cypher_question(self):
        f = open(self.CYPHER_PATH, "w")
        for i in self.var_list:
            output = cypher_answer_template % i['name'] + '\n'
            print(output)
            f.write(output)



'''Q1 '''
ENGLISH_PATH = 'data/questions/english/english_q1.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q1.txt'


english_question_template = "What is the minimum temperature to grow {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN a.name, c.name, temperature.min;"


cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()



'''Q2'''

ENGLISH_PATH = 'data/questions/english/english_q2.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q2.txt'


english_question_template = "What is the maximum temperature to grow {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()


'''Q3'''

ENGLISH_PATH = 'data/questions/english/english_q3.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q3.txt'

english_question_template = "At what range of temperature will {name} grow?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, temperature.min, temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q4'''

ENGLISH_PATH = 'data/questions/english/english_q4.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q4.txt'

english_question_template = "What is the optimal maximum temperature to grow {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (optimal)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q5'''

ENGLISH_PATH = 'data/questions/english/english_q5.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q5.txt'

english_question_template = "What is the optimal minimum temperature to grow {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (optimal)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, temperature.min;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q6'''

ENGLISH_PATH = 'data/questions/english/english_q6.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q6.txt'

english_question_template = "What is the optimal range of temperature to grow {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (optimal)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, temperature.min, temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q7'''

ENGLISH_PATH = 'data/questions/english/english_q7.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q7.txt'

english_question_template = "What kind of plant is {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(description:description) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

"""Q8"""

ENGLISH_PATH = 'data/questions/english/english_q8.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q8.txt'

english_question_template = "Describe the physiology of {name}."

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(description:description) \
RETURN  a.name, c.name, description.physiology;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

"""Q9"""

ENGLISH_PATH = 'data/questions/english/english_q9.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q9.txt'

english_question_template = "Describe the habit of {name}."

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(description:description) \
RETURN  a.name, c.name, description.habit;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

"""Q10"""

ENGLISH_PATH = 'data/questions/english/english_q10.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q10.txt'

english_question_template = "Describe the life form of {name}."

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(description:description) \
RETURN  a.name, c.name, description.life_form;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

"""Q11"""

ENGLISH_PATH = 'data/questions/english/english_q11.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q11.txt'

english_question_template = "What is the crop cycle of {name} like?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(cultivation:cultivation) \
MATCH (cultivation)-[:has]->(crop_cycle:crop_cycle) \
RETURN  a.name, c.name, crop_cycle.crop_cycle_min, crop_cycle.crop_cycle_max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q12 '''
ENGLISH_PATH = 'data/questions/english/english_q12.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q12.txt'


english_question_template = "What is the minimum soil pH to grow {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (absolute)-[:consist_of]->(soil_ph_absolute:soil_ph_absolute) \
RETURN  a.name, c.name, soil_ph_absolute.min;"


cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q13 '''
ENGLISH_PATH = 'data/questions/english/english_q13.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q13.txt'


english_question_template = "What is the maximum soil pH to grow {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (absolute)-[:consist_of]->(soil_ph_absolute:soil_ph_absolute) \
RETURN  a.name, c.name, soil_ph_absolute.max;"


cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q14 '''
ENGLISH_PATH = 'data/questions/english/english_q14.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q14.txt'


english_question_template = "At what range of soil pH will {name} grow?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (absolute)-[:consist_of]->(soil_ph_absolute:soil_ph_absolute) \
RETURN  a.name, c.name, soil_ph_absolute.min, soil_ph_absolute.max;"


cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q15 '''
ENGLISH_PATH = 'data/questions/english/english_q15.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q15.txt'


english_question_template = "What is the minimum optimal soil pH to grow {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
RETURN  a.name, c.name, soil_ph_optimal.min;"


cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q16 '''

ENGLISH_PATH = 'data/questions/english/english_q16.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q16.txt'


english_question_template = "What is the maximum optimal soil pH to grow {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
RETURN  a.name, c.name, soil_ph_optimal.max;"


cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q17 '''

ENGLISH_PATH = 'data/questions/english/english_q17.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q17.txt'


english_question_template = "What is the scientific name of {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
RETURN  a.name, c.name;"


cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q18 '''

ENGLISH_PATH = 'data/questions/english/english_q18.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q18.txt'


english_question_template = "Tell me everything about {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"


cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q19'''

ENGLISH_PATH = 'data/questions/english/english_q19.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q19.txt'


english_question_template = "Tell me the light intensity required to grow {name}."

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (ecology)-[:grows_in]->(o:optimal) \
MATCH (o)-[:consist_of]->(light:light_intensity_optimal) \
RETURN light.min, light.max;"


cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q20 '''

ENGLISH_PATH = 'data/questions/english/english_q20.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q20.txt'


english_question_template = "What is the biology of {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q21 '''

ENGLISH_PATH = 'data/questions/english/english_q21.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q21.txt'


english_question_template = "Please tell me everything about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q22 '''

ENGLISH_PATH = 'data/questions/english/english_q22.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q22.txt'


english_question_template = "I would like to know about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q23 '''

ENGLISH_PATH = 'data/questions/english/english_q23.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q23.txt'


english_question_template = "I would like to learn about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q24 '''

ENGLISH_PATH = 'data/questions/english/english_q24.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q24.txt'


english_question_template = "Can I get all the information about {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q25 '''

ENGLISH_PATH = 'data/questions/english/english_q25.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q25.txt'

english_question_template = "Can I get all the information about {name}?"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q26 '''

ENGLISH_PATH = 'data/questions/english/english_q26.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q26.txt'

english_question_template = "Please share with me everything about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q27 '''

ENGLISH_PATH = 'data/questions/english/english_q27.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q27.txt'

english_question_template = "I would like to know everything about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q28 '''

ENGLISH_PATH = 'data/questions/english/english_q28.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q28.txt'

english_question_template = "Give me a complete overview about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q29 '''

ENGLISH_PATH = 'data/questions/english/english_q29.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q29.txt'

english_question_template = "Give me detailed information about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q30 '''

ENGLISH_PATH = 'data/questions/english/english_q30.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q30.txt'

english_question_template = "Share with me everything about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q31 '''

ENGLISH_PATH = 'data/questions/english/english_q31.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q31.txt'

english_question_template = "Provide detailed information about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q32 '''

ENGLISH_PATH = 'data/questions/english/english_q31.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q31.txt'

english_question_template = "Provide detailed information about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q33 '''

ENGLISH_PATH = 'data/questions/english/english_q33.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q33.txt'

english_question_template = "Give me a full description about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q34 '''

ENGLISH_PATH = 'data/questions/english/english_q34.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q34.txt'

english_question_template = "Please provide a full description about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q35 '''

ENGLISH_PATH = 'data/questions/english/english_q35.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q35.txt'

english_question_template = "Give me a comprehensive description about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q36 '''

ENGLISH_PATH = 'data/questions/english/english_q36.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q36.txt'

english_question_template = "Please provide a comprehensive description about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q37 '''

ENGLISH_PATH = 'data/questions/english/english_q37.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q37.txt'

english_question_template = "Give me a comprehensive description about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q38 '''

ENGLISH_PATH = 'data/questions/english/english_q38.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q38.txt'

english_question_template = "Give me an in-depth description about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q39 '''

ENGLISH_PATH = 'data/questions/english/english_q39.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q39.txt'

english_question_template = "Tell me the in-depth description about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q40 '''

ENGLISH_PATH = 'data/questions/english/english_q40.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q40.txt'

english_question_template = "I want to know everything about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()

'''Q41 '''

ENGLISH_PATH = 'data/questions/english/english_q41.txt'
CYPHER_PATH = 'data/questions/cypher/cypher_q41.txt'

english_question_template = "List all the things you know about {name}"

cypher_question_template = "MATCH (c:crop_alias) return c.name as name;"

cypher_answer_template = "MATCH (a:crop_alias {name: '%s'})-[:is_alias_of]->(c:crop) \
MATCH (c)-[:has]->(ecology) \
MATCH (c)-[:has]->(description:description) \
MATCH (ecology)-[:grows_in]->(optimal) \
MATCH (ecology)-[:grows_in]->(absolute) \
MATCH (optimal)-[:consist_of]->(soil_ph_optimal:soil_ph_optimal) \
MATCH (absolute)-[:consist_of]->(temperature:temperature_required_optimal) \
RETURN  a.name, c.name, description.habit, description.life_form, description.physiology, \
soil_ph_optimal.max, soil_ph_optimal.min, temperature.min, \
temperature.max;"

cypherquestion = CypherQuestion(english_question_template, cypher_question_template, cypher_answer_template,
                                ENGLISH_PATH, CYPHER_PATH)

cypherquestion.generate_english_question()

cypherquestion.generate_cypher_question()