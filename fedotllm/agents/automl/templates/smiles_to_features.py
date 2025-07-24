def smiles_to_features(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        features = {
            {%descriptors%}
        }
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
        fp = mfpgen.GetFingerprint(mol)
        features.update({f'FP_{i}': int(b) for i, b in enumerate(fp)})
        
        return features
    except:
        return None