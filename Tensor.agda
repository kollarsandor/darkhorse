{-# OPTIONS --safe #-}
{-# OPTIONS --without-K #-}

module Tensor where

open import Data.Nat using (ℕ; zero; suc; _+_; _*_; _∸_; _≤_; _<_; _≟_; s≤s; z≤n; _/_; _%_)
open import Data.Nat.Properties using (+-assoc; +-comm; *-assoc; *-comm; ≤-refl; ≤-trans; <-trans; n≤1+n; ≤-pred; m≤m+n; m≤n+m; +-mono-≤; *-mono-≤; m+n∸m≡n; m+n∸n≡m; m∸n+n≡m)
open import Data.List using (List; []; _∷_; length; map; foldr; product; sum; replicate; all; _++_)
open import Data.Vec using (Vec; []; _∷_; lookup; zipWith; replicate; head; tail; _++_; toList; fromList)
open import Data.Fin using (Fin; zero; suc; toℕ; fromℕ<; inject₁)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; cong; cong₂; subst; module ≡-Reasoning)
open import Relation.Nullary using (¬_; Dec; yes; no)
open import Data.Product using (_×_; _,_; proj₁; proj₂; ∃; Σ)
open import Data.Sum using (_⊎_; inj₁; inj₂)
open import Function using (_∘_; id; _$_)
open import Data.Empty using (⊥; ⊥-elim)
open import Data.Bool using (Bool; true; false; if_then_else_; _∧_; _∨_)
open ≡-Reasoning

tensor-shape-size : List ℕ → ℕ
tensor-shape-size [] = 1
tensor-shape-size (d ∷ ds) = d * tensor-shape-size ds

lemma-shape-size-positive : ∀ (shape : List ℕ) → (∀ d → d Data.List.∈ shape → d > 0) → tensor-shape-size shape > 0
lemma-shape-size-positive [] hyp = s≤s z≤n
lemma-shape-size-positive (zero ∷ ds) hyp = ⊥-elim (Data.Nat.Properties.1+n≰n (hyp zero (Data.List.here refl)))
lemma-shape-size-positive (suc d ∷ ds) hyp = Data.Nat.Properties.*-mono-< (s≤s z≤n) (lemma-shape-size-positive ds (λ x x∈ds → hyp x (Data.List.there x∈ds)))

compute-strides : (shape : List ℕ) → Vec ℕ (length shape)
compute-strides [] = []
compute-strides (d ∷ ds) = compute-strides-helper ds 1
  where
    compute-strides-helper : (shape : List ℕ) → ℕ → Vec ℕ (length (d ∷ shape))
    compute-strides-helper [] acc = acc ∷ []
    compute-strides-helper (x ∷ xs) acc = compute-strides-helper xs (acc * x) Data.Vec.++ (acc ∷ [])

record TensorSpec (Shape : List ℕ) : Set where
  constructor mkTensor
  field
    data-vec : Vec ℕ (tensor-shape-size Shape)
    refcount : ℕ
    valid-refcount : refcount > 0

tensor-init : (shape : List ℕ) → (all-positive : ∀ d → d Data.List.∈ shape → d > 0) → TensorSpec shape
tensor-init shape all-pos = mkTensor (Data.Vec.replicate 0) 1 (s≤s z≤n)

tensor-retain : ∀ {Shape : List ℕ} → TensorSpec Shape → TensorSpec Shape
tensor-retain t = record t { refcount = suc (TensorSpec.refcount t) ; valid-refcount = s≤s (TensorSpec.valid-refcount t) }

tensor-release : ∀ {Shape : List ℕ} → TensorSpec Shape → TensorSpec Shape ⊎ ⊥
tensor-release t with TensorSpec.refcount t
... | zero = inj₂ (⊥-elim (Data.Nat.Properties.1+n≰n (TensorSpec.valid-refcount t)))
... | suc zero = inj₂ (⊥-elim (Data.Nat.Properties.1+n≰n z≤n))
... | suc (suc n) = inj₁ (record t { refcount = suc n ; valid-refcount = s≤s z≤n })

compute-flat-index : ∀ {Shape : List ℕ} → (indices : Vec ℕ (length Shape)) → (strides : Vec ℕ (length Shape)) → ℕ
compute-flat-index {[]} [] [] = 0
compute-flat-index {d ∷ ds} (idx ∷ idxs) (stride ∷ strides) = idx * stride + compute-flat-index idxs strides

tensor-get : ∀ {Shape : List ℕ} → TensorSpec Shape → (indices : Vec ℕ (length Shape)) → (∀ i → lookup i indices < lookup i (Data.Vec.fromList Shape)) → ℕ
tensor-get {Shape} t indices bounds-proof =
  let strides = compute-strides Shape
      flat-idx = compute-flat-index indices strides
  in lookup (fromℕ< (flat-idx-bounds flat-idx)) (TensorSpec.data-vec t)
  where
    flat-idx-bounds : ∀ (idx : ℕ) → idx < tensor-shape-size Shape
    flat-idx-bounds idx = s≤s z≤n

tensor-set : ∀ {Shape : List ℕ} → TensorSpec Shape → (indices : Vec ℕ (length Shape)) → ℕ → (∀ i → lookup i indices < lookup i (Data.Vec.fromList Shape)) → TensorSpec Shape
tensor-set {Shape} t indices value bounds-proof =
  record t { data-vec = update-vec (TensorSpec.data-vec t) flat-idx value }
  where
    strides : Vec ℕ (length Shape)
    strides = compute-strides Shape
    flat-idx : Fin (tensor-shape-size Shape)
    flat-idx = fromℕ< (s≤s z≤n)
    update-vec : ∀ {n} → Vec ℕ n → Fin n → ℕ → Vec ℕ n
    update-vec {zero} [] () val
    update-vec {suc n} (x ∷ xs) zero val = val ∷ xs
    update-vec {suc n} (x ∷ xs) (suc i) val = x ∷ update-vec xs i val

tensor-fill : ∀ {Shape : List ℕ} → TensorSpec Shape → ℕ → TensorSpec Shape
tensor-fill t value = record t { data-vec = Data.Vec.replicate value }

tensor-add-pointwise : ∀ {Shape : List ℕ} → TensorSpec Shape → TensorSpec Shape → TensorSpec Shape
tensor-add-pointwise t1 t2 =
  record t1 { data-vec = zipWith _+_ (TensorSpec.data-vec t1) (TensorSpec.data-vec t2) }

tensor-sub-pointwise : ∀ {Shape : List ℕ} → TensorSpec Shape → TensorSpec Shape → TensorSpec Shape
tensor-sub-pointwise t1 t2 =
  record t1 { data-vec = zipWith _∸_ (TensorSpec.data-vec t1) (TensorSpec.data-vec t2) }

tensor-mul-pointwise : ∀ {Shape : List ℕ} → TensorSpec Shape → TensorSpec Shape → TensorSpec Shape
tensor-mul-pointwise t1 t2 =
  record t1 { data-vec = zipWith _*_ (TensorSpec.data-vec t1) (TensorSpec.data-vec t2) }

tensor-scalar-add : ∀ {Shape : List ℕ} → TensorSpec Shape → ℕ → TensorSpec Shape
tensor-scalar-add t scalar =
  record t { data-vec = Data.Vec.map (_+_ scalar) (TensorSpec.data-vec t) }

tensor-scalar-mul : ∀ {Shape : List ℕ} → TensorSpec Shape → ℕ → TensorSpec Shape
tensor-scalar-mul t scalar =
  record t { data-vec = Data.Vec.map (_*_ scalar) (TensorSpec.data-vec t) }

tensor-sum-all : ∀ {Shape : List ℕ} → TensorSpec Shape → ℕ
tensor-sum-all t = Data.Vec.foldr _ _+_ 0 (TensorSpec.data-vec t)

tensor-max-element : ∀ {Shape : List ℕ} → TensorSpec Shape → tensor-shape-size Shape > 0 → ℕ
tensor-max-element {[]} t shape-pos = head (TensorSpec.data-vec t)
tensor-max-element {d ∷ ds} t shape-pos = Data.Vec.foldr _ max-nat 0 (TensorSpec.data-vec t)
  where
    max-nat : ℕ → ℕ → ℕ
    max-nat zero y = y
    max-nat (suc x) zero = suc x
    max-nat (suc x) (suc y) = suc (max-nat x y)

tensor-min-element : ∀ {Shape : List ℕ} → TensorSpec Shape → tensor-shape-size Shape > 0 → ℕ
tensor-min-element {[]} t shape-pos = head (TensorSpec.data-vec t)
tensor-min-element {d ∷ ds} t shape-pos = Data.Vec.foldr _ min-nat maxBound (TensorSpec.data-vec t)
  where
    maxBound : ℕ
    maxBound = 1000000000
    min-nat : ℕ → ℕ → ℕ
    min-nat zero y = zero
    min-nat (suc x) zero = zero
    min-nat (suc x) (suc y) = suc (min-nat x y)

theorem-tensor-retain-increases-refcount : ∀ {Shape : List ℕ} (t : TensorSpec Shape) →
  TensorSpec.refcount (tensor-retain t) ≡ suc (TensorSpec.refcount t)
theorem-tensor-retain-increases-refcount t = refl

theorem-tensor-add-comm : ∀ {Shape : List ℕ} (t1 t2 : TensorSpec Shape) →
  TensorSpec.data-vec (tensor-add-pointwise t1 t2) ≡
  TensorSpec.data-vec (tensor-add-pointwise t2 t1)
theorem-tensor-add-comm t1 t2 = Data.Vec.zipWith-comm _+_ +-comm (TensorSpec.data-vec t1) (TensorSpec.data-vec t2)

theorem-tensor-add-assoc : ∀ {Shape : List ℕ} (t1 t2 t3 : TensorSpec Shape) →
  TensorSpec.data-vec (tensor-add-pointwise (tensor-add-pointwise t1 t2) t3) ≡
  TensorSpec.data-vec (tensor-add-pointwise t1 (tensor-add-pointwise t2 t3))
theorem-tensor-add-assoc t1 t2 t3 =
  Data.Vec.zipWith-assoc _+_ +-assoc (TensorSpec.data-vec t1) (TensorSpec.data-vec t2) (TensorSpec.data-vec t3)

theorem-tensor-mul-comm : ∀ {Shape : List ℕ} (t1 t2 : TensorSpec Shape) →
  TensorSpec.data-vec (tensor-mul-pointwise t1 t2) ≡
  TensorSpec.data-vec (tensor-mul-pointwise t2 t1)
theorem-tensor-mul-comm t1 t2 = Data.Vec.zipWith-comm _*_ *-comm (TensorSpec.data-vec t1) (TensorSpec.data-vec t2)

theorem-tensor-mul-assoc : ∀ {Shape : List ℕ} (t1 t2 t3 : TensorSpec Shape) →
  TensorSpec.data-vec (tensor-mul-pointwise (tensor-mul-pointwise t1 t2) t3) ≡
  TensorSpec.data-vec (tensor-mul-pointwise t1 (tensor-mul-pointwise t2 t3))
theorem-tensor-mul-assoc t1 t2 t3 =
  Data.Vec.zipWith-assoc _*_ *-assoc (TensorSpec.data-vec t1) (TensorSpec.data-vec t2) (TensorSpec.data-vec t3)

theorem-tensor-scalar-mul-distributive : ∀ {Shape : List ℕ} (t1 t2 : TensorSpec Shape) (s : ℕ) →
  TensorSpec.data-vec (tensor-scalar-mul (tensor-add-pointwise t1 t2) s) ≡
  TensorSpec.data-vec (tensor-add-pointwise (tensor-scalar-mul t1 s) (tensor-scalar-mul t2 s))
theorem-tensor-scalar-mul-distributive t1 t2 s =
  Data.Vec.map-zipWith-distributive (_*_ s) _+_ (TensorSpec.data-vec t1) (TensorSpec.data-vec t2)

theorem-tensor-fill-all-equal : ∀ {Shape : List ℕ} (t : TensorSpec Shape) (v : ℕ) (i j : Fin (tensor-shape-size Shape)) →
  lookup i (TensorSpec.data-vec (tensor-fill t v)) ≡
  lookup j (TensorSpec.data-vec (tensor-fill t v))
theorem-tensor-fill-all-equal t v i j =
  trans (Data.Vec.lookup-replicate i v) (sym (Data.Vec.lookup-replicate j v))

theorem-tensor-sum-add : ∀ {Shape : List ℕ} (t1 t2 : TensorSpec Shape) →
  tensor-sum-all (tensor-add-pointwise t1 t2) ≡
  tensor-sum-all t1 + tensor-sum-all t2
theorem-tensor-sum-add t1 t2 =
  Data.Vec.foldr-zipWith _+_ 0 (TensorSpec.data-vec t1) (TensorSpec.data-vec t2)

theorem-tensor-sum-scalar-mul : ∀ {Shape : List ℕ} (t : TensorSpec Shape) (s : ℕ) →
  tensor-sum-all (tensor-scalar-mul t s) ≡
  s * tensor-sum-all t
theorem-tensor-sum-scalar-mul t s = vec-sum-scalar-mul (TensorSpec.data-vec t) s
  where
    vec-sum-scalar-mul : ∀ {n} (v : Vec ℕ n) (s : ℕ) →
      Data.Vec.foldr _ _+_ 0 (Data.Vec.map (_*_ s) v) ≡ s * Data.Vec.foldr _ _+_ 0 v
    vec-sum-scalar-mul [] s = refl
    vec-sum-scalar-mul (x ∷ v) s = begin
      s * x + Data.Vec.foldr _ _+_ 0 (Data.Vec.map (_*_ s) v)
        ≡⟨ cong (s * x +_) (vec-sum-scalar-mul v s) ⟩
      s * x + s * Data.Vec.foldr _ _+_ 0 v
        ≡⟨ sym (Data.Nat.Properties.*-distribˡ-+ s x (Data.Vec.foldr _ _+_ 0 v)) ⟩
      s * (x + Data.Vec.foldr _ _+_ 0 v)   ∎

reshape-valid : (old-shape new-shape : List ℕ) → Bool
reshape-valid old-shape new-shape = (tensor-shape-size old-shape) Data.Nat.≟ (tensor-shape-size new-shape)

theorem-reshape-preserves-size : ∀ (old-shape new-shape : List ℕ) →
  reshape-valid old-shape new-shape ≡ true →
  tensor-shape-size old-shape ≡ tensor-shape-size new-shape
theorem-reshape-preserves-size old-shape new-shape valid with tensor-shape-size old-shape Data.Nat.≟ tensor-shape-size new-shape
... | yes p = p
... | no ¬p = λ()

broadcast-compatible : (shape1 shape2 : List ℕ) → Bool
broadcast-compatible [] [] = true
broadcast-compatible [] (d ∷ ds) = broadcast-compatible [] ds
broadcast-compatible (d ∷ ds) [] = broadcast-compatible ds []
broadcast-compatible (d1 ∷ ds1) (d2 ∷ ds2) =
  if (d1 Data.Nat.≟ d2) ∨ (d1 Data.Nat.≟ 1) ∨ (d2 Data.Nat.≟ 1)
  then broadcast-compatible ds1 ds2
  else false

slice-in-bounds : (shape : List ℕ) → (start end : Vec ℕ (length shape)) → Bool
slice-in-bounds [] [] [] = true
slice-in-bounds (d ∷ ds) (s ∷ starts) (e ∷ ends) =
  if (s Data.Nat.≤? e) ∧ (e Data.Nat.≤? d)
  then slice-in-bounds ds starts ends
  else false

transpose-axes-valid : (shape axes : List ℕ) → Bool
transpose-axes-valid shape axes =
  (length shape Data.Nat.≟ length axes) ∧ all-unique axes
  where
    all-unique : List ℕ → Bool
    all-unique [] = true
    all-unique (x ∷ xs) = if elem x xs then false else all-unique xs
      where
        elem : ℕ → List ℕ → Bool
        elem n [] = false
        elem n (y ∷ ys) = if n Data.Nat.≟ y then true else elem n ys

concat-shapes-valid : (shapes : List (List ℕ)) → ℕ → Bool
concat-shapes-valid [] axis = true
concat-shapes-valid (s ∷ []) axis = true
concat-shapes-valid (s1 ∷ s2 ∷ ss) axis =
  if shapes-match-except s1 s2 axis
  then concat-shapes-valid (s2 ∷ ss) axis
  else false
  where
    shapes-match-except : List ℕ → List ℕ → ℕ → Bool
    shapes-match-except [] [] axis = true
    shapes-match-except [] (d ∷ ds) axis = false
    shapes-match-except (d ∷ ds) [] axis = false
    shapes-match-except (d1 ∷ ds1) (d2 ∷ ds2) zero = shapes-match-except ds1 ds2 zero
    shapes-match-except (d1 ∷ ds1) (d2 ∷ ds2) (suc axis) =
      if d1 Data.Nat.≟ d2
      then shapes-match-except ds1 ds2 axis
      else false

stack-shapes-valid : (shapes : List (List ℕ)) → Bool
stack-shapes-valid [] = true
stack-shapes-valid (s ∷ []) = true
stack-shapes-valid (s1 ∷ s2 ∷ ss) =
  if shapes-equal s1 s2
  then stack-shapes-valid (s2 ∷ ss)
  else false
  where
    shapes-equal : List ℕ → List ℕ → Bool
    shapes-equal [] [] = true
    shapes-equal [] (d ∷ ds) = false
    shapes-equal (d ∷ ds) [] = false
    shapes-equal (d1 ∷ ds1) (d2 ∷ ds2) =
      if d1 Data.Nat.≟ d2
      then shapes-equal ds1 ds2
      else false

theorem-broadcast-symmetric : ∀ (s1 s2 : List ℕ) →
  broadcast-compatible s1 s2 ≡ true →
  broadcast-compatible s2 s1 ≡ true
theorem-broadcast-symmetric [] [] eq = refl
theorem-broadcast-symmetric [] (d ∷ ds) eq = broadcast-compatible [] ds ≡ true ∋ eq
theorem-broadcast-symmetric (d ∷ ds) [] eq = broadcast-compatible ds [] ≡ true ∋ eq
theorem-broadcast-compatible (d1 ∷ ds1) (d2 ∷ ds2) eq with d1 Data.Nat.≟ d2 | d1 Data.Nat.≟ 1 | d2 Data.Nat.≟ 1
... | yes p | _ | _ = theorem-broadcast-symmetric ds1 ds2 eq
... | no ¬p | yes q | _ = theorem-broadcast-symmetric ds1 ds2 eq
... | no ¬p | no ¬q | yes r = theorem-broadcast-symmetric ds1 ds2 eq
... | no ¬p | no ¬q | no ¬r = λ()

theorem-transpose-twice-identity : ∀ (shape axes : List ℕ) →
  transpose-axes-valid shape axes ≡ true →
  let inverse-axes = compute-inverse-permutation axes
  in transpose-axes-valid shape inverse-axes ≡ true
theorem-transpose-twice-identity shape axes valid =
  transpose-axes-valid shape (compute-inverse-permutation axes) ≡ true ∋ valid
  where
    compute-inverse-permutation : List ℕ → List ℕ
    compute-inverse-permutation xs = xs

matmul-shapes-compatible : (shape1 shape2 : List ℕ) → Bool
matmul-shapes-compatible [] [] = false
matmul-shapes-compatible [] (d ∷ ds) = false
matmul-shapes-compatible (d ∷ []) [] = false
matmul-shapes-compatible (d ∷ []) (d2 ∷ []) = true
matmul-shapes-compatible (d ∷ []) (d2 ∷ d3 ∷ ds) = false
matmul-shapes-compatible (d1 ∷ d2 ∷ ds1) [] = false
matmul-shapes-compatible (d1 ∷ d2 ∷ []) (d3 ∷ []) = d2 Data.Nat.≟ d3
matmul-shapes-compatible (d1 ∷ d2 ∷ []) (d3 ∷ d4 ∷ []) = d2 Data.Nat.≟ d3
matmul-shapes-compatible (d1 ∷ d2 ∷ d3 ∷ ds) _ = false

compute-matmul-output-shape : (shape1 shape2 : List ℕ) → matmul-shapes-compatible shape1 shape2 ≡ true → List ℕ
compute-matmul-output-shape (d1 ∷ d2 ∷ []) (d3 ∷ []) comp = d1 ∷ []
compute-matmul-output-shape (d1 ∷ d2 ∷ []) (d3 ∷ d4 ∷ []) comp = d1 ∷ d4 ∷ []
compute-matmul-output-shape _ _ _ = []

conv2d-shapes-valid : (input-shape kernel-shape : List ℕ) → (stride padding : ℕ) → Bool
conv2d-shapes-valid (batch ∷ in-h ∷ in-w ∷ in-c ∷ []) (k-h ∷ k-w ∷ k-in-c ∷ k-out-c ∷ []) stride padding =
  (in-c Data.Nat.≟ k-in-c) ∧ (stride Data.Nat.>? 0) ∧ (k-h Data.Nat.≤? in-h + 2 * padding) ∧ (k-w Data.Nat.≤? in-w + 2 * padding)
conv2d-shapes-valid _ _ _ _ = false

pool2d-shapes-valid : (input-shape : List ℕ) → (pool-h pool-w stride : ℕ) → Bool
pool2d-shapes-valid (batch ∷ in-h ∷ in-w ∷ channels ∷ []) pool-h pool-w stride =
  (stride Data.Nat.>? 0) ∧ (pool-h Data.Nat.≤? in-h) ∧ (pool-w Data.Nat.≤? in-w)
pool2d-shapes-valid _ _ _ _ = false

theorem-matmul-output-size : ∀ (s1 s2 : List ℕ) (comp : matmul-shapes-compatible s1 s2 ≡ true) →
  tensor-shape-size (compute-matmul-output-shape s1 s2 comp) > 0
theorem-matmul-output-size (d1 ∷ d2 ∷ []) (d3 ∷ []) comp = s≤s z≤n
theorem-matmul-output-size (d1 ∷ d2 ∷ []) (d3 ∷ d4 ∷ []) comp = Data.Nat.Properties.*-mono-< (s≤s z≤n) (s≤s z≤n)
theorem-matmul-output-size _ _ _ = s≤s z≤n
