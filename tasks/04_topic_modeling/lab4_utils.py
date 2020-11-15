import os
import numpy as np


def load_collection(path, verbose=True):
    """
    Load collection of document from disk
    
    Arguments
    ---------
    path : str
        Path to folder with txt files
    
    verbose : bool
        If True then print additional info 
    
    Return
    ------
    collection : dict (key=file name (str), value=text collection (str))
    """
    collection = dict()
    for file_name in os.listdir(path):
        with open(path + '/' + file_name, 'r') as f_input:
            new_document = f_input.read()
            collection[file_name] = new_document
    
    if verbose:
        print('Total number of documents: {}'.format(len(collection)))
        print()
        print('Some document examples: ')
        document_examples = list(collection.values())[:5]
        document_examples = [example[:100] + '...' for example in document_examples]
        for example in document_examples:
            print('\t' + example)
        print()
        print('Some file names examples: ')
        file_name_examples = list(collection.keys())[:5]
        for example in file_name_examples:
            print('\t' + example)
            

    return collection


def load_parallel_documents_info(path, verbose=True):
    """
    Read files with information about parallel documents
    
    Arguments
    ---------
    path : str
        Path to file
        
    Return
    ------
    parallel_info : dict (key=file name of one language(str),
                          value=file name of other language (str))
    """
    parallel_info = dict()
    with open(path, 'r') as f:
        for line in f:
            one_language_doc, other_language_doc = line.split()
            parallel_info[one_language_doc.strip()] = other_language_doc.strip()
    if verbose:
        print('Total number of pairs: {}'.format(len(parallel_info)))
    return parallel_info


def write_vw_lab4(output_path,
                  en_cleaned_collection,
                  ru_cleaned_collection,
                  es_cleaned_collection,
                  en_ru_parallel_docs,
                  en_es_parallel_docs,
                  use_parallel_info):
    """
    Write collection on a disk in vowpal wabbit format.
    
    Arguments
    ---------
    output_path : str
        Path of output file
    
    en_cleaned_collection : dict (key=file name (str), value=text document (str))
        Collection of cleaned russian texts
    
    ru_cleaned_collection : dict (key=file name (str), value=text document (str))
        Collection of cleaned russian texts
        
    es_cleaned_collection : dict (key=file name (str), value=text document (str))
        Collection of cleaned spanish texts
        
    en_ru_parallel_docs : dict (key=file name of one language(str),
                                value=file name of other language (str))
        Parallel info for english and russian
        
    en_es_parallel_docs :  dict (key=file name of one language(str),
                                 value=file name of other language (str))
        Parallel info for english and spanish
    
    use_parallel_info : bool
        If True then parallel docs info is used to costruct vowpal wabbit file.
    """    
    with open(output_path, 'w') as f_output:
        # en documents and parallel info
        for en_file_name, en_content in en_cleaned_collection.items():
            new_document = '{} |@english {} '.format(en_file_name, en_content)
            
            if use_parallel_info:
                if en_file_name in en_ru_parallel_docs:
                    ru_content = ru_cleaned_collection[en_ru_parallel_docs[en_file_name]]
                    new_document += '|@russian {} '.format(ru_content)

                if en_file_name in en_es_parallel_docs:
                    es_content = es_cleaned_collection[en_es_parallel_docs[en_file_name]]
                    new_document += '|@spanish {}'.format(es_content)
            
            f_output.write(new_document + '\n')
        
        for ru_file_name, ru_content in ru_cleaned_collection.items():
            if use_parallel_info:
                if ru_file_name in en_ru_parallel_docs.values():
                    continue
            new_document = '{} |@russian {}'.format(ru_file_name, ru_content)
            f_output.write(new_document + '\n')
        
        for es_file_name, es_content in es_cleaned_collection.items():
            if use_parallel_info:
                if es_file_name in en_es_parallel_docs.values():
                    continue
            new_document = '{} |@spanish {}'.format(es_file_name, es_content)
            f_output.write(new_document + '\n')        
        

def get_indexes_of_relevant_documents(theta, parallel_docs, metric='cosine'):
    """
    Calculate position in search output
    for each translation document in parallel_docs.  

    Parameters
    ----------
    theta : pandas.DataFrame of shape (n_topics, n_documents)
    
    parallel_docs : dict (key=file name of one language(str),
                          value=file name of other language (str))
        Parallel info for two languages
    
    metric : str, cosine or dot
        Metric for neighbors calculating

    Return
    -------
    result_index : list of int
        Positions of each document in search output

    docs_pair : list of (str, str)
        Sequence of docs pairs from parallel_docs.
        docs_pair[i] corresponds to result_index[i]. 
    """
    if metric not in {'cosine', 'dot'}:
        raise TypeError('metric should be cosine or dot')
    
    theta_T = theta.T
    embeddigns_norms = (theta_T ** 2).sum(axis=1) ** 0.5
        
    one_language_relevant_docs, _ = zip(*parallel_docs.items())
    one_language_relevant_docs = list(one_language_relevant_docs)
    one_language_relevant_embeddings = (
        theta_T
        .loc[one_language_relevant_docs]
    )
    one_language_relevant_norms = (
        embeddigns_norms
        .loc[one_language_relevant_docs]
    )
    
    second_language_example = next(iter(parallel_docs.values()))
    second_language_docs = [
        document
        for document in theta_T.index
        if document[:2] == second_language_example[:2]
    ]
    second_language_embeddings = theta_T.loc[second_language_docs]
    second_language_norms = embeddigns_norms.loc[second_language_docs]
    
    similarity_matrix = (
        one_language_relevant_embeddings
        .dot(second_language_embeddings.T)
    )
    
    if metric == 'cosine':
        similarity_matrix = similarity_matrix.div(one_language_relevant_norms, axis='index')
        similarity_matrix = similarity_matrix.div(second_language_norms, axis='columns')
    
    similarity_matrix_array = similarity_matrix.values
    estimations = similarity_matrix_array.argsort(axis=1)[:, ::-1]
    second_lang_doc_to_number = dict(zip(similarity_matrix.columns,
                                         range(len(similarity_matrix.columns))))
    
    result_index = []
    docs_pair = []
    
    for i, one_document in enumerate(similarity_matrix.index):
        if parallel_docs[one_document] not in similarity_matrix.columns:
            continue
        doc_index = second_lang_doc_to_number[
            parallel_docs[one_document]
        ]
        index_in_output = np.where(estimations[i] == doc_index)[0]
        result_index.append(index_in_output[0])
        docs_pair.append((one_document, parallel_docs[one_document]))
    return result_index, docs_pair