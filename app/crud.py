from typing import List, Dict, Any, Union
from bson import ObjectId

from . import schemas
from .database import get_collection

# Get the MongoDB collection for analysis results
analysis_results_collection = get_collection("analysis_results")

def get_analysis_result(result_id: str) -> Union[Dict[str, Any], None]:
    try:
        # MongoDB uses _id for the primary key
        result = analysis_results_collection.find_one({"_id": ObjectId(result_id)})
        if result:
            # Convert ObjectId to string for JSON serialization
            result["id"] = str(result["_id"])
            return result
    except Exception:
        return None
    return None

def get_analysis_results(skip: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
    results = []
    for result in analysis_results_collection.find().skip(skip).limit(limit):
        result["id"] = str(result["_id"])
        results.append(result)
    return results

def create_analysis_result(result: schemas.AnalysisResultCreate) -> Dict[str, Any]:
    # Convert Pydantic model to dictionary for MongoDB insertion
    result_dict = result.model_dump(by_alias=True, exclude_unset=True)
    inserted_result = analysis_results_collection.insert_one(result_dict)
    
    # Retrieve the inserted document to return it with the generated _id
    new_result = analysis_results_collection.find_one({"_id": inserted_result.inserted_id})
    if new_result:
        new_result["id"] = str(new_result["_id"])
        return new_result
    return {}
