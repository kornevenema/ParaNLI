# ParaNLI
Repository for the Thesis of Information Science at the University in Groningen. For our thesis, we will paraphrase the MNLI dataset, specifically the validation set. We will compare the performance of the RoBERTa-large-MNLI on the original and the paraphrased dataset.


##ToDo
- Create paraphrased dataset for each of the following paraphraser:
  - BART Paraphrase Model (Lewis et al., 2019)
  - PEGASUS paraphraser (J. Zhang et al., 2020)
  - Parrot paraphraser (Damodaran, 2021)
  - chatGPT API
- Test paraphraser performer using:
  - UniEval score
  - BERTScore
  - BLEU score
- Create one paraphrased dataset using the different models based on the 
  evaluation scores.
- Manually annotate 5% of this dataset.
- Test performance of this dataset on the RoBERTa-large-MNLI model.
- Compare its performance between the original and paraphrased dataset.
- Select 25 random examples and investigate patterns and common mistakes. 


##Licensing Information
The majority of the corpus is released under the OANC’s license, which allows all content to be freely used, modified, and shared under permissive terms. The data in the FICTION section falls under several permissive licenses; Seven Swords is available under a Creative Commons Share-Alike 3.0 Unported License, and with the explicit permission of the author, Living History and Password Incorrect are available under Creative Commons Attribution 3.0 Unported Licenses; the remaining works of fiction are in the public domain in the United States (but may be licensed differently elsewhere).