# OPSD EMA vs No-GT EMA vs Frozen SFT Teacher: MMStar Answer Format Examples

Date: 2026-06-08

## Setup

- Dataset: MMStar
- Ratio: 30%
- EMA model: `opsd_qwen25vl_opsd_ema_full512_r030`
- No-ground-truth EMA model: `opsd_qwen25vl_opsd_nogt_ema_full512_r030`
- Frozen/SFT teacher OPSD model: `opsd_qwen25vl_opsd_sftteacher_full512_r030`
- EMA checkpoint family: `opsd_ema_alpha0p01_full_18204_ddp4_512`
- No-ground-truth EMA checkpoint family: `opsd_nogt_ema_alpha0p01_full_18204_ddp4_512`
- Frozen/SFT teacher checkpoint family: `opsd_sft_teacher_full_18204_ddp4_512`

## Answer Format Pattern

### EMA teacher OPSD

The EMA model often outputs a teacher-prompt-contaminated explanation instead of a concise multiple-choice answer. Common patterns:

```text
The problem asks us to derive ...
The reference solution reasoning ...
Therefore, the answer is A
---
Here
```

Observed issues:

- It often does not end with a clean `A/B/C/D`.
- It often mentions `reference solution reasoning`, which should not appear at evaluation time.
- It sometimes contains the correct option in a sentence but fails exact matching.
- It sometimes explicitly chooses the wrong option.
- Some generations are truncated mid-sentence.

### No-ground-truth EMA teacher OPSD

The no-ground-truth EMA model does not show the same `reference solution reasoning` leak in these examples, but it still tends to answer with long prose and often fails to produce a clean final option.

Observed issues:

- It often starts a general explanation instead of emitting only the option.
- It often truncates before a final `A/B/C/D` answer.
- It sometimes chooses or implies a wrong option.
- In these 10 examples, all no-ground-truth EMA generations are marked `hit=0`.

### Frozen/SFT teacher OPSD

The frozen/SFT teacher model usually follows a clean answer format:

```text
A. The suitcase is on the book.
D. Skyscrapers
C
B
```

Observed behavior:

- It usually emits a concise option letter.
- Often includes the option text after the letter.
- It does not leak the training teacher prompt style.

## Examples

### 1. Index 0: Format failure despite correct content

Question: Which option describe the object relationship in the image correctly?

Ground truth: A

Options:

- A. The suitcase is on the book.
- B. The suitcase is beneath the cat.
- C. The suitcase is beneath the bed.
- D. The suitcase is beneath the book.

EMA answer:

```text
---

The problem asks us to derive an understanding that the suitcase is on top of the book, and since it is clear that the suitcase is on the book,
```

No-ground-truth EMA answer:

```text
The correct answer is A. The suitcase is on top of the car trunk, not underneath it. Therefore, the correct option is:

A. The suitcase is
```

Frozen/SFT teacher answer:

```text
A. The suitcase is on the book.
```

### 2. Index 32: Answer appears but not clean final option

Question: What is the predominant feature in the image?

Ground truth: D

Options:

- A. Cars
- B. People
- C. Buildings
- D. Skyscrapers

EMA answer:

```text
The problem asks us to derive that the tall buildings are the dominant feature, and since they are the tall buildings, the answer is D. The reference solution reasoning
```

No-ground-truth EMA answer:

```text
The most prominent feature in this image is the illuminated skyscrapers and buildings in the city, which dominate the scene and draw the viewer's attention. The vibrant
```

Frozen/SFT teacher answer:

```text
D. Skyscrapers
```

### 3. Index 253: Truncated reasoning, no option

Question: Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What is the sum of 2002, 2003 and 2004?

Ground truth: C

Options:

- A. 40.7
- B. 74
- C. 70.4
- D. 70.0

EMA answer:

```text
The problem asks for the sum of the percentages in 2002, 2003, and 2004. Using my own
```

No-ground-truth EMA answer:

```text
To determine the sum of 2002, 2003 and 2004, we need to add these three years together:
```

Frozen/SFT teacher answer:

```text
C
```

### 4. Index 505: Format failure despite correct content

Question: Which of the following statements match the image?

Ground truth: D

Options:

- A. A green pentagon is above a red shape.
- B. A red ellipse is above a green pentagon.
- C. A yellow shape is below a red pentagon.
- D. A pentagon is below a pentagon.

EMA answer:

```text
The problem asks us to derive that a pentagon is below another pentagon, and since there is a reference solution where it is derived that the pentagon is
```

No-ground-truth EMA answer:

```text
To determine which statement matches the image, let's analyze the image step by step:

1. **Identify the shapes in the image:**
   - The
```

Frozen/SFT teacher answer:

```text
D. A pentagon is below a pentagon.
```

### 5. Index 84: Explicit wrong option

Question: What is the primary focus of the people in the image?

Ground truth: D

Options:

- A. Watching a performance
- B. Standing in lines for a concert
- C. Participating in a protest
- D. Walking down the street

EMA answer:

```text
The reference solution to this problem is clear and straightforward. The people are watching a performance, so they must be watching a performance. Therefore, the answer is A
```

No-ground-truth EMA answer:

```text
The most likely answer to what the photo depicts is that it shows a man and woman walking on a path near a park, with trees and buildings in the background
```

Frozen/SFT teacher answer:

```text
D. Walking down the street
```

### 6. Index 258: Explicit wrong option

Question: What color are the walls in the bathroom?

Ground truth: C

Options:

- A. White
- B. Blue
- C. Beige
- D. Tan

EMA answer:

```text
The problem asks you to derive that the walls are white, and since they are white, the same answer is A. The reference solution reasoning is that the walls
```

No-ground-truth EMA answer:

```text
The correct answer is A. White

The bathroom has white tiles on the floor and walls, which suggests that the walls are most likely painted white. Therefore,
```

Frozen/SFT teacher answer:

```text
C. Beige
```

### 7. Index 310: Explicit wrong option

Question: How many people are present in the image?

Ground truth: B

Options:

- A. 1
- B. 2
- C. 4
- D. 3

EMA answer:

```text
The problem asks us to derive an understanding that there is one person on the reference solution, and since there is only one person, the answer is A. Here
```

No-ground-truth EMA answer:

```text
The image shows a man walking on a catwalk during a fashion show. The man is wearing a blue and white striped shirt, and he is walking on a
```

Frozen/SFT teacher answer:

```text
B
```

### 8. Index 355: Explicit wrong option

Question: What is the shape of the object with an angel face and wings in the image?

Ground truth: A

Options:

- A. Egg-shaped
- B. Square
- C. Round
- D. Star-shaped

EMA answer:

```text
The problem asks us to derive that the object, which is an egg-shaped item, is round and thus the same solution to this problem is C.

---

Here
```

No-ground-truth EMA answer:

```text
The image shows a white, round object with a shiny surface, which suggests it could be a balloon or a balloon. Given that balloons often come in various shapes
```

Frozen/SFT teacher answer:

```text
A. Egg-shaped
```

### 9. Index 656: Explicit wrong option

Question: What is the predominant color of the sky in the image?

Ground truth: B

Options:

- A. Blue
- B. Gray
- C. Yellow
- D. Orange

EMA answer:

```text
The problem asks us to derive that the sky is orange and thus the same solution is D. The reference solution reasoning is that the sky is orange, so it
```

No-ground-truth EMA answer:

```text
The most likely color of the horizon in this image is blue, as it is a common color for the horizon in many landscapes, including mountains and hills, which
```

Frozen/SFT teacher answer:

```text
B. Gray
```

### 10. Index 1275: Explicit wrong option

Question: The organelle labeled D is called

Ground truth: A

Options:

- A. Cell membrane
- B. Cytoplasm
- C. Nuclear membrane
- D. Cytoskeleton

EMA answer:

```text
The problem asks you to derive that the organelle labeled as D is the nuclear membrane, and since it is the nuclear membrane, the answer is C.
```

No-ground-truth EMA answer:

```text


The image shows a plant growing in a pot, with a black background. The plant has a green stem and a brownish-green leaf. The
```

Frozen/SFT teacher answer:

```text
A. Cell membrane
```
