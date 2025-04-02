; Symptoms with attributes (inspired by MedlinePlus descriptions)
!(add-atom &self (Symptom ((fever ((description "Elevated body temperature") (common "yes") (severity "moderate")))
                          (cough ((description "Expelling air from lungs") (common "yes") (severity "low")))
                          (headache ((description "Pain in the head") (common "yes") (severity "moderate")))
                          (fatigue ((description "Extreme tiredness") (common "yes") (severity "high"))))))

; Diseases with attributes
!(add-atom &self (Disease ((flu ((description "Viral infection causing fever and cough") (severity "moderate") (contagious "yes")))
                          (migraine ((description "Severe headache with nausea") (severity "high") (contagious "no")))
                          (depression ((description "Mental health condition causing fatigue") (severity "high") (contagious "no"))))))

; Treatments with attributes
!(add-atom &self (Treatment ((flu ((description "rest, fluids, antiviral drugs") (duration "5-7 days")))
                            (migraine ((description "pain relievers, hydration, dark room") (duration "1-3 days")))
                            (depression ((description "therapy, antidepressants") (duration "ongoing"))))))

!(add-atom &self (fever have-relation flu))
!(add-atom &self (cough have-relation flu))
!(add-atom &self (headache have-relation migraine))
!(add-atom &self (fatigue have-relation depression))

!(add-atom &self (faq (("What are the symptoms of the flu?" ((answer "The flu often causes fever, cough, sore throat, runny or stuffy nose, body aches, headache, chills, and fatigue. Some people may also have vomiting or diarrhea, though this is more common in children.")))
                      ("How can I treat a headache at home?" ((answer "For a mild headache, try resting in a quiet, dark room, drinking plenty of water, and taking over-the-counter pain relievers like ibuprofen or acetaminophen. If it persists, consult a doctor.")))
                      ("When should I see a doctor for a fever?" ((answer "See a doctor if your fever is 103°F (39.4°C) or higher, lasts more than 3 days, or comes with severe symptoms like trouble breathing, chest pain, or confusion.")))
                      ("Can fatigue be a sign of something serious?" ((answer "Yes, fatigue can be a symptom of conditions like depression, anemia, or infections. If it lasts for weeks or you feel unusually weak, talk to a healthcare provider.")))
                      ("How do I know if I have a migraine or just a regular headache?" ((answer "Migraines often include throbbing pain on one side of the head, nausea, and sensitivity to light or sound. Regular headaches are usually less severe and don’t have these extra symptoms.")))
                      ("Hi" ((answer "Hello! How can I assist you with your health today?"))))))