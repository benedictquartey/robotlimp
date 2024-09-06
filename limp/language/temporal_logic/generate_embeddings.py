from tqdm import tqdm
import openai
import pickle

if __name__ == "__main__":
    embeddings_path = "./embedding_cache" #path to where embeddings are saved
    dataset_dir = "./ltl_datasets" #path to where the dataset is saved

    dataset_name = "small-droneplanning" #dataset name

    english_txt = "eng.txt" #name of the english txt file
    ltl_txt = "ltl.txt" #name of the ltl txt file

    dataset_path = f"{dataset_dir}/{dataset_name}" #path to where the dataset is saved
    embeddings_path = f"{embeddings_path}/{dataset_name}" #path to where embeddings will be saved

    lang2embeddings = {}  #dictionary to store language to embeddings
    lang2ltl = {} #dictionary to store language to ltl

    #the english_txt and ltl_txt are files containing the english and ltl sentences respectively. We first need to read them in together
    with open(f"{dataset_path}/{english_txt}") as f:
        contents = f.read()
        english_list = contents.split("\n")

    with open(f"{dataset_path}/{ltl_txt}") as f:
        contents = f.read()
        ltl_list = contents.split("\n")

    #zip the two lists together
    langltl_list = list(zip(english_list,ltl_list))

    #loop through the list and generate embeddings for each language
    for lang, ltl in tqdm(langltl_list):
        #generate the embedding
        embedding = openai.Embedding.create(input=lang, model="text-embedding-ada-002")["data"][0]["embedding"]
        #add the embedding to the dictionary
        lang2embeddings[lang] = embedding
        lang2ltl[lang] = ltl

    #save the lang2embeddings dictionary
    with open(f"{embeddings_path}_lang2embeddings.pkl", "wb") as f:
        pickle.dump(lang2embeddings, f)

    #save the lang2ltl dictionary
    with open(f"{embeddings_path}_lang2ltl.pkl", "wb") as f:
        pickle.dump(lang2ltl, f)
