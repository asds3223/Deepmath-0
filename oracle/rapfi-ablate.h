// ablate.h — (Deepmath-0 오라클 전용 수정) Rapfi 탐색 기능별 on/off 스위치. 기본값은 원래 동작.
#pragma once
namespace Search::AB::Ablate {
inline bool vcfLeaf = true, razor = true, futility = true, nullMove = true, lmp = true, oppo5Ext = true,
            singular = true, lmr = true, failHigh = true, defendGen = true, iirTT = true;
}
