import torch
import copy

from src.data.vocabulary import BOS, EOS, PAD
from src.models.base import NMTModel
from .utils import mask_scores, tensor_gather_helper


def fixwords_beam_search(fw_nmt_model, bw_nmt_model, beam_size, max_steps, src_seqs, alpha, constrains, positions, last_sentences, imt_step=1, bidirection = False):
    batch_size = len(constrains)

    patience = len(constrains[0])

    fw_enc_outputs = fw_nmt_model.encode(src_seqs)
    bw_enc_outputs = bw_nmt_model.encode(src_seqs)



    total_scores = None
    total_ids = None
    total_lengths = None

    for p in range(patience):
        last_sents = copy.deepcopy(last_sentences)
        min_pos = [max_steps for ii in range(batch_size)]
        max_pos = [0 for ii in range(batch_size)]
        for b in range(batch_size):
            for idx in range(imt_step):
                pos = positions[b][p][idx]
                cons = constrains[b][p][idx]
                if pos < len(last_sents[b]):
                    last_sents[b][pos] = cons
                else:
                    last_sents[b].append(cons)
                if min_pos[b] > pos:
                    min_pos[b] = pos
                if max_pos[b] < pos:
                    max_pos[b] = pos

        fw_init_dec_states = fw_nmt_model.init_decoder(fw_enc_outputs, expand_size=beam_size)

        beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
        final_lengths = src_seqs.new(batch_size, beam_size).zero_().float()
        beam_scores = src_seqs.new(batch_size, beam_size).zero_().float()
        final_word_indices = src_seqs.new(batch_size, beam_size, 1).fill_(BOS)
        dec_states = fw_init_dec_states

        for t in range(max_steps):
            next_scores, dec_states = fw_nmt_model.decode(final_word_indices.view(batch_size * beam_size, -1), dec_states)

            next_scores = - next_scores  # convert to negative log_probs
            next_scores = next_scores.view(batch_size, beam_size, -1)
            next_scores = mask_scores(scores=next_scores, beam_mask=beam_mask)

            beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

            vocab_size = beam_scores.size(-1)



            # Length penalty
            if alpha > 0.0:
                normed_scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + beam_mask + final_lengths).unsqueeze(
                    2) ** alpha
            else:
                normed_scores = beam_scores.detach().clone()

            for b in range(batch_size):
                pos = max_pos[b]
                if t <= pos and t < len(last_sents[b]):
                    fixword = last_sents[b][t]
                    normed_scores[b, :, fixword] = float('-inf')
                if t == pos + 1:
                    normed_scores[b, 1:, :] = float('inf')

            normed_scores = normed_scores.view(batch_size, -1)



            # Get topK with beams
            # indices: [batch_size, ]
            _, indices = torch.topk(normed_scores, k=beam_size, dim=-1, largest=False, sorted=False)
            next_beam_ids = torch.div(indices, vocab_size)  # [batch_size, ]
            next_word_ids = indices % vocab_size  # [batch_size, ]

            # Re-arrange by new beam indices
            beam_scores = beam_scores.view(batch_size, -1)
            beam_scores = torch.gather(beam_scores, 1, indices)

            beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                             gather_from=beam_mask,
                                             batch_size=batch_size,
                                             beam_size=beam_size,
                                             gather_shape=[-1])

            final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                      gather_from=final_word_indices,
                                                      batch_size=batch_size,
                                                      beam_size=beam_size,
                                                      gather_shape=[batch_size * beam_size, -1])

            final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                                 gather_from=final_lengths,
                                                 batch_size=batch_size,
                                                 beam_size=beam_size,
                                                 gather_shape=[-1])

            dec_states = fw_nmt_model.reorder_dec_states(dec_states, new_beam_indices=next_beam_ids, beam_size=beam_size)

            # If next_word_ids is EOS, beam_mask_ should be 0.0
            beam_mask_ = 1.0 - next_word_ids.eq(EOS).float()
            next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                       PAD)  # If last step a EOS is already generated, we replace the last token as PAD
            beam_mask = beam_mask * beam_mask_

            # # If an EOS or PAD is encountered, set the beam mask to 0.0
            final_lengths += beam_mask

            final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)
            if beam_mask.eq(0.0).all():
                break

        if alpha > 0.0:
            scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + final_lengths) ** alpha
        else:
            scores = beam_scores / final_lengths

        _, reranked_ids = torch.sort(scores, dim=-1, descending=False)

        final_word_indices = tensor_gather_helper(gather_indices=reranked_ids,
                                    gather_from=final_word_indices[:, :, 1:].contiguous(),
                                    batch_size=batch_size,
                                    beam_size=beam_size,
                                    gather_shape=[batch_size * beam_size, -1])

        if bidirection:
            fixed_word_ids = final_word_indices.clone()
            fixed_word_ids = fixed_word_ids[:, 0, :]
            bw_positions = [0 for ii in range(batch_size)]
            for ia in range(batch_size):
                ic = 0
                while ic < len(fixed_word_ids[ia]) and fixed_word_ids[ia, ic] != EOS:
                    ic += 1
                bw_positions[ia] = ic
                ic -= 1
                for id in range(int(int(ic) / 2) + 1):
                    tmp = int(fixed_word_ids[ia, id])
                    fixed_word_ids[ia, id] = int(fixed_word_ids[ia, ic - id])
                    fixed_word_ids[ia, ic - id] = tmp


            for b in range(batch_size):
                min_pos[b] = bw_positions[b] - min_pos[b] - 1
                for idx in range(imt_step):
                    positions[b][p][idx] = bw_positions[b] - positions[b][p][idx] -1
            # backward

            bw_init_dec_states = bw_nmt_model.init_decoder(bw_enc_outputs, expand_size=beam_size)

            beam_mask = src_seqs.new(batch_size, beam_size).fill_(1).float()
            final_lengths = src_seqs.new(batch_size, beam_size).zero_().float()
            beam_scores = src_seqs.new(batch_size, beam_size).zero_().float()
            final_word_indices = src_seqs.new(batch_size, beam_size, 1).fill_(BOS)
            dec_states = bw_init_dec_states

            for t in range(max_steps):
                next_scores, dec_states = bw_nmt_model.decode(final_word_indices.view(batch_size * beam_size, -1),
                                                              dec_states)

                next_scores = - next_scores  # convert to negative log_probs
                next_scores = next_scores.view(batch_size, beam_size, -1)
                next_scores = mask_scores(scores=next_scores, beam_mask=beam_mask)

                beam_scores = next_scores + beam_scores.unsqueeze(2)  # [B, Bm, N] + [B, Bm, 1] ==> [B, Bm, N]

                vocab_size = beam_scores.size(-1)

                # Length penalty
                if alpha > 0.0:
                    normed_scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + beam_mask + final_lengths).unsqueeze(
                        2) ** alpha
                else:
                    normed_scores = beam_scores.detach().clone()

                for b in range(batch_size):
                    pos = min_pos[b]
                    if t <= pos:
                        fixword = fixed_word_ids[b][t]
                        normed_scores[b, :, fixword] = float('-inf')
                    if t == pos + 1:
                        normed_scores[b, 1:, :] = float('inf')

                normed_scores = normed_scores.view(batch_size, -1)

                # Get topK with beams
                # indices: [batch_size, ]
                _, indices = torch.topk(normed_scores, k=beam_size, dim=-1, largest=False, sorted=False)
                next_beam_ids = torch.div(indices, vocab_size)  # [batch_size, ]
                next_word_ids = indices % vocab_size  # [batch_size, ]

                # Re-arrange by new beam indices
                beam_scores = beam_scores.view(batch_size, -1)
                beam_scores = torch.gather(beam_scores, 1, indices)

                beam_mask = tensor_gather_helper(gather_indices=next_beam_ids,
                                                 gather_from=beam_mask,
                                                 batch_size=batch_size,
                                                 beam_size=beam_size,
                                                 gather_shape=[-1])

                final_word_indices = tensor_gather_helper(gather_indices=next_beam_ids,
                                                          gather_from=final_word_indices,
                                                          batch_size=batch_size,
                                                          beam_size=beam_size,
                                                          gather_shape=[batch_size * beam_size, -1])

                final_lengths = tensor_gather_helper(gather_indices=next_beam_ids,
                                                     gather_from=final_lengths,
                                                     batch_size=batch_size,
                                                     beam_size=beam_size,
                                                     gather_shape=[-1])

                dec_states = bw_nmt_model.reorder_dec_states(dec_states, new_beam_indices=next_beam_ids,
                                                             beam_size=beam_size)

                # If next_word_ids is EOS, beam_mask_ should be 0.0
                beam_mask_ = 1.0 - next_word_ids.eq(EOS).float()
                next_word_ids.masked_fill_((beam_mask_ + beam_mask).eq(0.0),
                                           PAD)  # If last step a EOS is already generated, we replace the last token as PAD
                beam_mask = beam_mask * beam_mask_

                # # If an EOS or PAD is encountered, set the beam mask to 0.0
                final_lengths += beam_mask

                final_word_indices = torch.cat((final_word_indices, next_word_ids.unsqueeze(2)), dim=2)
                if beam_mask.eq(0.0).all():
                    break



            final_word_indices = final_word_indices[:, :, 1:].contiguous()
            bw_positions = [0 for ii in range(batch_size)]
            for ia in range(batch_size):
                for ib in range(beam_size):
                    ic = 0
                    while ic < len(final_word_indices[ia, ib]) and final_word_indices[ia, ib, ic] != EOS:
                        ic += 1
                    if ib == 0:
                        bw_positions[ia] = ic
                    ic -= 1
                    for id in range(int(int(ic) / 2) + 1):
                        tmp = int(final_word_indices[ia, ib, id])
                        final_word_indices[ia, ib, id] = int(final_word_indices[ia, ib, ic - id])
                        final_word_indices[ia, ib, ic - id] = tmp

            for b in range(batch_size):
                for idx in range(imt_step):
                    positions[b][p][idx] = bw_positions[b] - positions[b][p][idx] - 1

            if alpha > 0.0:
                scores = beam_scores * (5.0 + 1.0) ** alpha / (5.0 + final_lengths) ** alpha
            else:
                scores = beam_scores / final_lengths

            _, reranked_ids = torch.sort(scores, dim=-1, descending=False)
            final_word_indices = tensor_gather_helper(gather_indices=reranked_ids,
                                 gather_from=final_word_indices,
                                 batch_size=batch_size,
                                 beam_size=beam_size,
                                 gather_shape=[batch_size * beam_size, -1])

        if p == 0:
            total_scores = beam_scores.clone()
            total_ids = final_word_indices.clone()
            total_lengths = final_lengths.clone()
        else:
            total_scores = torch.cat((total_scores, beam_scores), dim=1)
            t_size = total_ids.size(-1)
            f_size = final_word_indices.size(-1)
            if t_size < f_size:
                pad_ids = src_seqs.new(batch_size, total_ids.size(1), f_size - t_size).fill_(PAD)
                total_ids = torch.cat((total_ids, pad_ids), dim=2)
            elif t_size > f_size:
                pad_ids = src_seqs.new(batch_size, beam_size, t_size - f_size).fill_(PAD)
                final_word_indices = torch.cat((final_word_indices, pad_ids), dim=2)
            total_ids = torch.cat((total_ids, final_word_indices), dim=1)
            total_lengths = torch.cat((total_lengths, final_lengths), dim=1)



    return total_ids, positions
