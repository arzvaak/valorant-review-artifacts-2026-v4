# Data card and reconstruction scope

## Purpose and units

Schedule-based retrospective forecasting of canonical Team A map wins after picks and series wins before play. Stable team identity fixes orientation before target attachment. A map is one archived Game ID; a series requires the complete archived map inventory and a strict majority of valid map winners. Administrative awards, remakes and forfeits have no uniformly validated marker and are not inferred from scores.

The broad nonglobal primary scores cover 55,943 maps in 1,049 events and 25,186 series in 1,047 events. The reviewed six regional circuits (24,254 maps, 10,515 series; 340 events) are an overlapping secondary scope. Known globals (859 maps, 336 series; 14 events) are separate evaluation-only records. The matched 2025 diagnostic is a subset, not an independent population. The later unreviewed continuation has 1,613 scored maps and 734 scored series in 23 events and is not verified professional-only or prospective evidence. Known China events and protected confirmation are excluded; events with unknown classification remain explicit.

## Sources, time and measurement

Immutable VLR.gg-derived archival records supply source-bound identities, scheduled anchors, map/agent lineups and earlier performance observations. Display names are historical aliases, never unrestricted numeric predictors. The map horizon assumes archived observed map/players/agents are known after picks. The series horizon reconstructs earlier-roster and preferred-agent proxies; it never substitutes actual future players or picks. Actual publication timestamps and starting lineups at the earlier series horizon are unverified.

Original core history releases at scheduled anchor +2h and predictions use anchor -15min. The online extension uses +6h. Separate exploratory six-hour/30-minute and 24-hour/60-minute schedule-clock comparisons retain the map architecture and Elo-versus-context distinction (Supplement S22), without verifying actual publication times or separating delay from lead. Equal-time forecasts precede summed updates; IDs do not order equal-time maps.

## Inputs and missingness

Shared B has eight numeric columns and already contains Elo/map-history summaries. N adds 228 raw numeric columns across nine semantic families; L adds 22 linked columns. Composition/map channels and player tensors have fold-local vocabularies. Field counts are not independent mechanisms. Dictionaries specify transformations, swap mappings, denominators and missing masks. Current-target performance, scores, rounds and actual future series picks are prohibited predictors.

Coverage is computed on the exact scored broad nonglobal rows. `core-coverage.csv` gives count, denominator and fraction for both-team history, roster-slot presence, player exposure, linked ACS and training-vocabulary support. Map roster presence is archival observed-slot completeness, not verified preplay publication. Series presence is a historical proxy. Linked ACS completeness is one concrete statistic-specific measure, not complete coverage of every linked variable.

## Partitions and decisions

Twelve rolling whole-event windows separate training, early stopping, selection, calibration and evaluation. Match IDs stay intact. Events may reappear as training history in a later window but each outer target is evaluated once. `core-phase-counts.csv` records actual events/rows/series at each stage, including separate evaluation-only globals. The article lists observed date spans. All core histories admit only block training observations; later phases never update them. Online and later-feedback policies are separate.

## Shared and restricted artifacts

The v4.2 software and aggregate companion contains source code, feature dictionaries, schemas, chronological split and cutoff logic, recorded environments and configurations, grouped results, and figure inputs. It adds coverage and phase tables, a bounded implementation audit, and event-level forecast-loss summaries. The earlier v4.1 release is preserved unchanged. Aggregate reconstruction does not retrain forecasting models.

Raw captures, source-derived row tables, row-level labels and forecasts, source identity keys, fitted models and checkpoints, and protected records are not part of the redistributable software and aggregate materials. The MIT and CC BY 4.0 licenses apply to project-authored software, documentation, and aggregates only to the extent the project has rights to share them; they do not grant rights to the underlying third-party source records. This does not claim that source terms prohibit every possible form of redistribution. The release includes neither an unrestricted source-row license nor a complete model-refit dataset.

Original row-level predictions are retained in the restricted local research store with input/method/code identities. They are not represented as publicly downloadable. Reviewers can inspect the shared event-level summaries and reconstruction code; independent row-level rescoring/refitting additionally requires compatible lawful inputs and a separate permitted access arrangement. No publication timestamp, raw record or model binary is fabricated to fill that gap.

## Interpretation and intended use

Event-equal Brier/log loss are primary; ordinary accuracy is row-weighted and separately labelled. Complete-event intervals condition on saved fits; they do not include retraining/selection variation or fully resolve repeated-team dependence. Correction is within each stated comparison family, not across the whole adaptive research programme. Retrospective later performance, unreviewed classifications and hypothetical timing do not establish deployment readiness or betting profitability. The benchmark is intended to compare conditional information value and its limitations.
