'''
MB: This is the sample code from the BERTopic website with some
placeholder code for loading StackOverflow data. This currently
does both fitting and prediction, but we can split that into separate tasks.  
'''
import argparse
import joblib

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import xmltodict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process an input file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python main.py input.txt --verbose"
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input file"
    )

    return parser.parse_args()
    
'''
Download data, save to preferred location, reference 
'Comments.xml' data location in data load below.
'''

if __name__ == '__main__':
    args = parse_args()

    with open(args.input_file, 'r') as f:
        data = f.read()
    xml_dict = xmltodict.parse(data)
    documents = [d['@Text'] for d in xml_dict]

    # Pass it into BERTopic
    topic_model = joblib.load("bertopic_model.pkl")

    topics, probs = topic_model.transform(docs)
