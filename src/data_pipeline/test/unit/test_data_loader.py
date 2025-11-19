from exception import CGAgentException
import pandas as pd
import os,sys
import yaml
from pathlib import Path
import pytest
from src.data_pipeline.data_loader import Data_Reader

#fixture for temp files

@pytest.fixture
def temp_yaml_file(tmp_path):
    content= {"source_path":str(tmp_path/"sample.csv"),"raw_path":str(tmp_path/"raw.csv"),"dataset_name":"test_dataset"}
    yaml_path= tmp_path/"paths.yaml"
    with open(yaml_path,"w") as f:
        yaml.safe_dump(content,f)
    return yaml_path,content

@pytest.fixture
def temp_csv_file(tmp_path):
    df=pd.DataFrame({"A":[1,2,3],"B":[4,5,6]})
    csv_path= tmp_path/"sample.csv"
    df.to_csv(csv_path,index=False)
    return csv_path,df


@pytest.fixture
def temp_features_yaml(tmp_path):
    content= {"features":["A","B"],"target":"B"}
    features_path= tmp_path/"features.yaml"
    with open (features_path,"w") as f:
        yaml.safe_dump(content,f)
    return features_path,content


#test load yaml
def test_load_yaml(temp_yaml_file):
    yaml_path,content= temp_yaml_file
    loaded=Data_Reader.load_yaml(str(yaml_path))
    assert loaded== content

#test fielNotFound
def test_load_data_file_not_found(temp_features_yaml):
    paths_yaml="non_existent_file.yaml"
    features_yaml,_= temp_features_yaml

    with pytest.raises(CGAgentException):
        Data_Reader.load_data(paths_config=paths_yaml, features_config=str(features_yaml))

#test load data
def test_load_data(temp_csv_file,temp_yaml_file,temp_features_yaml):
    csv_path,df_original= temp_csv_file
    paths_yaml,paths_content= temp_yaml_file
    features_yaml,features_content=temp_features_yaml

    loaded_df,loaded_paths,loaded_features=Data_Reader.load_data(paths_config=str(paths_yaml),
                                                                     features_config=str(features_yaml))

    #chcek returned dataframe matches original
    pd.testing.assert_frame_equal(loaded_df,df_original)

    # Check configs loaded correctly
    assert loaded_paths == paths_content
    assert loaded_features == features_content
    
    #chcek csv saved to raw_path
    raw_path= Path(paths_content["raw_path"])
    assert raw_path.exists()
    saved_df= pd.read_csv(raw_path)
    pd.testing.assert_frame_equal(saved_df,df_original)

    pd.testing.assert_frame_equal(saved_df,df_original)
