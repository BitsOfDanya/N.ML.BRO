# -*- coding: utf-8 -*
#
# распечатаем мн-ва заболеваний-их категории в консоль, причем
# категории большими буквами, а диагнозы внутри категории -
# маленькими через запятую

import os
import json


def parse_diagnosis_hierarchy(diagnosis_hierarchy_json):
    """
    в файле-описании диагнозы группируются в категории, получается граф-дерево максимальной глубины 3 этажа (категория - категория - диагноз)
    :param diagnosis_hierarchy_json: файл с иерраохией
    :return: печатает в консоль чтоб оттуда было удобно копипастить
    """
    filename = os.fsdecode(diagnosis_hierarchy_json)
    if os.path.isfile(filename):
        with open(filename,  encoding='utf-8') as json_file:
            json_parsed = json.load(json_file)

            for unit_i in range(len(json_parsed)):
                if json_parsed[unit_i]['type'] == 'category':
                    category = json_parsed[unit_i]
                    print (category['name'].upper())
                    category_values = category['value']
                    diagnoses_in_category = ""
                    for value in category_values:
                        if value['type'] == 'diagnosis':
                            diagnoses_in_category+=value['name']+", "
                        else:
                            if  value['type'] == 'category':
                                diagnoses_in_category += value['name'].upper() + " ["
                                for diagnos in value['value']:
                                    if diagnos['type'] == 'diagnosis':
                                        diagnoses_in_category += diagnos['name']+", "
                                    else:
                                        print ("да твою мать какая тут вложенность то")
                                diagnoses_in_category+='] '
                    print (diagnoses_in_category)
                else:
                     if json_parsed[unit_i]['type'] == 'diagnosis':
                         print(json_parsed[unit_i]['name'] + " - диагноз вне категорий")
                     else:
                         print ("ERROR ?")







if __name__ == "__main__":
    path_to_summary = 'C:\\Users\\neuro\\Desktop\\diagnosis.json'  # кодировка utf-8 подразумевается
    parse_diagnosis_hierarchy(path_to_summary)


