from enum import Enum
from typing import Literal, Optional, Union, List

from fedot.core.repository.tasks import TaskTypesEnum
from pydantic import BaseModel, ConfigDict, Field


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TS_FORECASTING = "ts_forecasting"


class PresetType(str, Enum):
    BEST_QUALITY = "best_quality"
    FAST_TRAIN = "fast_train"
    STABLE = "stable"
    AUTO = "auto"
    GPU = "gpu"
    TS = "ts"
    AUTOML = "automl"


class ClassificationMetricsEnum(str, Enum):
    ROCAUC = "roc_auc"
    precision = "precision"
    # f1 = 'f1'
    # logloss = 'neg_log_loss'
    # ROCAUC_penalty = 'roc_auc_pen'
    accuracy = "accuracy"


class RegressionMetricsEnum(str, Enum):
    RMSE = "rmse"
    MSE = "mse"
    MSLE = "neg_mean_squared_log_error"
    MAPE = "mape"
    SMAPE = "smape"
    MAE = "mae"
    R2 = "r2"
    RMSE_penalty = "rmse_pen"


class TimeSeriesForecastingMetricsEnum(str, Enum):
    MASE = "mase"
    RMSE = "rmse"
    MSE = "mse"
    MSLE = "neg_mean_squared_log_error"
    MAPE = "mape"
    SMAPE = "smape"
    MAE = "mae"
    R2 = "r2"
    RMSE_penalty = "rmse_pen"


class FedotConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    problem: TaskTypesEnum = Field(
        ..., description="Name of the modelling problem to solve"
    )
    timeout: float = Field(
        ..., description="Time for model design (in minutes): Default: 1.0"
    )
    cv_folds: Optional[int] = Field(
        ..., description="Number of folds for cross-validation: Default: None"
    )
    preset: PresetType = Field(
        ...,
        description=(
            "Name of the preset for model building. Possible options:\n"
            "best_quality -> All models that are available for this data type and task are used\n"
            "fast_train -> Models that learn quickly. This includes preprocessing operations (data operations) that only reduce the dimensionality of the data, but cannot increase it. For example, there are no polynomial features and one-hot encoding operations\n"
            "stable -> The most reliable preset in which the most stable operations are included\n"
            "auto -> Automatically determine which preset should be used\n"
            "gpu -> Models that use GPU resources for computation\n"
            "ts -> A special preset with models for time series forecasting task\n"
            "automl -> A special preset with only AutoML libraries such as TPOT and H2O as operations"
            "Default: auto"
        ),
    )
    metric: Union[
        ClassificationMetricsEnum,
        RegressionMetricsEnum,
        TimeSeriesForecastingMetricsEnum,
    ] = Field(
        ...,
        description="Choose relevant to problem metric of model quality assessment.",
    )
    predict_method: Literal["predict", "predict_proba", "forecast"] = Field(
        ...,
        description="Method for prediction: predict - for classification and regression, predict_proba - for classification, forecast - for time series forecasting",
    )

#RDKit Descriptors

class RDKitDescriptorsEnum(str, Enum):
    MOLWT = "MolWt"
    HEAVYATOMMOLWT = "HeavyAtomMolWt"
    HEAVYATOMCOUNT = "HeavyAtomCount"
    NUMATOMS = "NumAtoms"
    NUMVALENCEELECTRONS = "NumValenceElectrons"

    # Lipophilicity/Hydrophobicity
    MOLLOGP = "MolLogP"
    MOLMR = "MolMR"

    # Hydrogen Bonding
    NUMHDONORS = "NumHDonors"
    NUMHACCEPTORS = "NumHAcceptors"

    # Topology and Connectivity
    TPSA = "TPSA"
    NUMROTATABLEBONDS = "NumRotatableBonds"
    RINGCOUNT = "RingCount"
    NUMAROMATICRINGS = "NumAromaticRings"
    NUMALIPHATICRINGS = "NumAliphaticRings"
    NUMSATURATEDRINGS = "NumSaturatedRings"
    NUMHETEROATOMS = "NumHeteroatoms"
    NUMAMIDEBONDS = "NumAmideBonds"

class RDKitConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    descriptors: List[RDKitDescriptorsEnum] = Field(
        ..., description=(
            """Here's a list of some of the most popular RDKit descriptors to choose from, with short explanations:

--Basic Molecular Properties--
MolWt -> Molecular Weight. The average molecular weight of the molecule (sum of atomic weights).
HeavyAtomMolWt -> Heavy Atom Molecular Weight. The molecular weight considering only non-hydrogen atoms.
HeavyAtomCount -> Number of Heavy Atoms. The count of non-hydrogen atoms in the molecule.
NumAtoms -> Number of Atoms. The total count of atoms in the molecule (including hydrogens if they are explicitly present).
NumValenceElectrons -> Number of Valence Electrons. The sum of valence electrons of all atoms in the molecule.

--Lipophilicity/Hydrophobicity--
MolLogP -> Molecular LogP (octanol-water partition coefficient). A measure of a molecule's lipophilicity, indicating its preference for a lipid (fat) environment over an aqueous (water) environment.
MolMR -> Molar Refractivity. A measure of the total polarizability of a molecule, related to its volume and electronic properties.

--Hydrogen Bonding--
NumHDonors -> Number of Hydrogen Bond Donors. The count of atoms capable of donating a hydrogen bond (typically N-H and O-H groups).
NumHAcceptors -> Number of Hydrogen Bond Acceptors. The count of atoms capable of accepting a hydrogen bond (typically O, N, F atoms with lone pairs).

--Topology and Connectivity--
TPSA -> Topological Polar Surface Area - The sum of polar surface areas of polar atoms (nitrogen, oxygen, and their attached hydrogens). It's a useful descriptor for predicting drug absorption and blood-brain barrier penetration.
NumRotatableBonds -> Number of Rotatable Bonds. The count of single bonds between two non-terminal heavy atoms, excluding amide C-N bonds and bonds to terminal acetylenes. This descriptor relates to molecular flexibility.
RingCount -> Total number of Rings.
NumAromaticRings -> Number of Aromatic Rings. The count of aromatic ring systems in the molecule.
NumAliphaticRings -> Number of Aliphatic Rings. The count of non-aromatic (aliphatic) ring systems.
NumSaturatedRings -> Number of Saturated Rings. The count of fully saturated ring systems.
NumHeteroatoms -> Number of Heteroatoms. The count of non-carbon and non-hydrogen atoms (e.g., O, N, S, P, halogens).
NumAmideBonds -> Number of Amide Bonds. The count of amide functional groups."""
        ),
    )