#!/usr/bin/env python3
# workflow.py - main workflow for running ml analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from ml_models import (
    ActivityPatternClassifier,
    SelfSupervisedActivityLearner,
    CurveSimilarityClassifier,
    EnsembleActivityClassifier,
    prepare_data_from_r,
)

def run_complete_ml_analysis(data_dir="ml_data", output_dir="ml_results"):
    print("loading data...")
    
    hourly_data, lit_labels = prepare_data_from_r(
        f"{data_dir}/hourly_activity_within.csv",
        f"{data_dir}/literature_labels.csv"
    )
    
    # optional metadata if exists
    try:
        metadata = pd.read_csv(f"{data_dir}/species_metadata.csv")
    except:
        metadata = None

    # create training strategies
    print("\ncreating training sets...")
    
    # strategy 1: literature labels
    print("  strategy 1: literature labels")
    lit_train = (lit_labels[lit_labels['confidence'] >= 3]
        .dropna(subset=['pattern'])          
        .copy()
    )
    lit_train['strategy'] = 'literature'
    print(f"  {len(lit_train)} high-confidence species")
    
    # strategy 2: self-supervised
    print("  strategy 2: self-supervised")
    ssl = SelfSupervisedActivityLearner()
    self_train = ssl.create_reference_patterns(hourly_data, min_confidence=0.85)
    self_train['strategy'] = 'self_supervised'
    print(f"  {len(self_train)} patterns identified")
    
    # strategy 3: mixed
    print("  strategy 3: mixed approach")
    mixed_train = lit_train.copy()
    new_species = self_train[~self_train['species'].isin(lit_train['species'])]
    mixed_train = (pd.concat([lit_train, new_species], ignore_index=True)
        .dropna(subset=['pattern'])       
    )
    mixed_train['strategy'] = 'mixed'
    print(f"  {len(mixed_train)} total training species")
    
    # save training sets
    Path(output_dir).mkdir(exist_ok=True)
    lit_train.to_csv(f"{output_dir}/training_literature.csv", index=False)
    self_train.to_csv(f"{output_dir}/training_self_supervised.csv", index=False)
    mixed_train.to_csv(f"{output_dir}/training_mixed.csv", index=False)
    
    # train models
    print("\ntraining models...")
    
    results = {}
    
    for strategy_name, train_data in [
        ('literature', lit_train),
        ('self_supervised', self_train),
        ('mixed', mixed_train)
    ]:
        print(f"\n  {strategy_name} strategy...")
        
        clf = ActivityPatternClassifier()
        features = clf.prepare_features(hourly_data)
        
        train_features = features[features['species'].isin(train_data['species'])]
        train_merged = train_features.merge(
            train_data[['species', 'pattern']], 
            on='species'
        )
        
        if len(train_merged) < 20:
            print(f"  skipping - too few examples ({len(train_merged)})")
            continue
        
        models = {}
        
        # logistic regression
        print("  training logistic regression...")
        lr_clf = ActivityPatternClassifier()
        lr_clf.train(train_merged.drop('pattern', axis=1), 
                    train_merged['pattern'], 
                    model_type='logistic')
        models['logistic'] = lr_clf
        
        # random forest
        print("  training random forest...")
        rf_clf = ActivityPatternClassifier()
        rf_clf.train(train_merged.drop('pattern', axis=1), 
                    train_merged['pattern'], 
                    model_type='random_forest')
        models['random_forest'] = rf_clf
        
        # ensemble
        print("  training ensemble...")
        ensemble = EnsembleActivityClassifier()
        ensemble.train_ensemble(hourly_data, train_data, include_self_supervised=False)
        models['ensemble'] = ensemble
        
        results[strategy_name] = {
            'train_data': train_data,
            'models': models,
            'n_train': len(train_merged)
        }
    
    # evaluate
    print("\nevaluating models...")
    
    eval_results = {}
    
    for strategy_name, strategy_results in results.items():
        print(f"\n  {strategy_name} strategy...")
        
        eval_species = lit_labels[lit_labels['confidence'] >= 3]['species'].tolist()
        eval_features = features[features['species'].isin(eval_species)]
        
        if len(eval_features) < 10:
            print("  skipping evaluation - too few examples")
            continue
        
        strategy_eval = {}
        
        for model_name, model in strategy_results['models'].items():
            if model_name == 'ensemble':
                preds = model.predict_ensemble(hourly_data)
                preds = preds[preds['species'].isin(eval_species)]
            else:
                preds = model.predict(eval_features)
            
            eval_df = preds.merge(
                lit_labels[['species', 'pattern']], 
                on='species'
            )
            
            pred_col = 'prediction' if 'prediction' in eval_df.columns else 'ensemble_prediction'
            accuracy = (eval_df[pred_col] == eval_df['pattern']).mean()
            
            strategy_eval[model_name] = {
                'accuracy': accuracy,
                'predictions': preds
            }
            
            print(f"    {model_name}: {accuracy:.3f}")
        
        eval_results[strategy_name] = strategy_eval
    
    # find best model
    print("\nanalyzing results...")
    
    best_strategy = max(eval_results.keys(), 
                       key=lambda k: max(eval_results[k][m]['accuracy'] 
                                       for m in eval_results[k]))
    best_model_name = max(eval_results[best_strategy].keys(), 
                         key=lambda k: eval_results[best_strategy][k]['accuracy'])
    
    print(f"  best: {best_strategy} - {best_model_name}")
    print(f"  accuracy: {eval_results[best_strategy][best_model_name]['accuracy']:.3f}")
    
    # save best model
    best_model = results[best_strategy]['models'][best_model_name]
    if hasattr(best_model, 'model'):
        joblib.dump(best_model, f"{output_dir}/best_model.pkl")
    
    # apply to all species
    print("\napplying to all species...")
    
    if best_model_name == 'ensemble':
        all_predictions = best_model.predict_ensemble(hourly_data)
    else:
        all_features = clf.prepare_features(hourly_data)
        all_predictions = best_model.predict(all_features)
    
    if metadata is not None:
        all_predictions = all_predictions.merge(metadata, on='species', how='left')
    
    all_predictions.to_csv(f"{output_dir}/all_predictions.csv", index=False)
    
    # create visualizations
    print("\ncreating visualizations...")
    
    # confusion matrix
    if len(eval_results) > 0:
        eval_df = eval_results[best_strategy][best_model_name]['predictions'].merge(
            lit_labels[['species', 'pattern']], on='species'
        )
        
        pred_col = 'prediction' if 'prediction' in eval_df.columns else 'ensemble_prediction'
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(eval_df['pattern'], eval_df[pred_col])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('predicted')
        plt.ylabel('true')
        plt.title(f'confusion matrix - {best_strategy} {best_model_name}')
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # feature importance
    if hasattr(best_model, 'feature_importance'):
        feat_imp = best_model.feature_importance()
        if feat_imp is not None:
            plt.figure(figsize=(10, 8))
            top_features = feat_imp.head(20)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('importance')
            plt.title('top 20 features')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nanalysis complete")
    print(f"results saved to {output_dir}/")
    
    return {
        'models': results,
        'evaluations': eval_results,
        'predictions': all_predictions,
        'best_model': (best_strategy, best_model_name)
    }


if __name__ == "__main__":
    results = run_complete_ml_analysis()
