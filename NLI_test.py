from transformers import pipeline
from PurposeReader import PurposeReader



if __name__ == "__main__":
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/nli-deberta-base')
    #scores = model.predict([("nir is great","someone else is bad")])

    label_mapping = ['contradiction', 'entailment', 'neutral']
    #labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

    dir = "1_milion_gpt3_tagged_patents-20240207T140452Z-001\\1_milion_gpt3_tagged_patents\\"
    PR = PurposeReader()
    purpose_dict = PR.create_purpose_dict(dir)
    with open("NLI_test_text.txt","w") as f:
        for key in list(purpose_dict.keys())[:1000]:
            for key2 in  list(purpose_dict.keys())[:1000]:
                if key != key2:
                    purp1 = purpose_dict[key]
                    purp2 = purpose_dict[key2]
                    scores = model.predict([purp1, purp2])
                    label = label_mapping[scores.argmax()]
                    if label == "entailment":
                        f.write(f"{purp1} ---> {purp2}\n")


