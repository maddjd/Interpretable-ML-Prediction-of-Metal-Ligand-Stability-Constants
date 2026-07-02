import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


class RDKit_2D:
    def __init__(self, smiles):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles

    def compute_2Drdkit(self, name, extra_columns):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(
            [x[0] for x in Descriptors._descList]
        )
        header = calc.GetDescriptorNames()

        for mol in self.mols:
            ds = calc.CalcDescriptors(mol)
            rdkit_2d_desc.append(ds)

        df = pd.DataFrame(rdkit_2d_desc, columns=header)
        df.insert(loc=0, column="smiles", value=self.smiles)

        for col_name, col_data in extra_columns.items():
            df[col_name] = col_data

        df.to_csv(name[:-4] + "_RDKit_2D.csv", index=False)


def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    input_filename = os.path.join(script_dir, "Dataset.csv")
    output_filename = os.path.join(script_dir, "Dataset_descriptors.csv")

    df = pd.read_csv(input_filename)
    smiles = df["smiles"].values

    extra_columns = {
        "logK": df[" logK"].values,
    }

    RDKit_descriptor = RDKit_2D(smiles)
    RDKit_descriptor.compute_2Drdkit(output_filename, extra_columns)


if __name__ == "__main__":
    main()