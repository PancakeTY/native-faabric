#include <faabric/batch-scheduler/RoundRobinScheduler.h>
#include <faabric/batch-scheduler/SchedulingDecision.h>
#include <faabric/util/batch.h>
#include <faabric/util/logging.h>
#include <algorithm>

namespace faabric::batch_scheduler {

static std::map<std::string, int> getHostFreqCount(
  std::shared_ptr<SchedulingDecision> decision)
{
    std::map<std::string, int> hostFreqCount;
    for (auto host : decision->hosts) {
        hostFreqCount[host] += 1;
    }

    return hostFreqCount;
}

// Given a new decision that improves on an old decision (i.e. to migrate), we
// want to make sure that we minimise the number of migration requests we send.
// This is, we want to keep as many host-message scheduling in the old decision
// as possible, and also have the overall locality of the new decision (i.e.
// the host-message histogram)
// NOTE: keep in mind that the newDecision has the right host histogram, but
// the messages may be completely out-of-order
static std::shared_ptr<SchedulingDecision> minimiseNumOfMigrations(
  std::shared_ptr<SchedulingDecision> newDecision,
  std::shared_ptr<SchedulingDecision> oldDecision)
{
    auto decision = std::make_shared<SchedulingDecision>(oldDecision->appId,
                                                         oldDecision->groupId);

    // We want to maintain the new decision's host-message histogram
    auto hostFreqCount = getHostFreqCount(newDecision);

    // Helper function to find the next host in the histogram with slots
    auto nextHostWithSlots = [&hostFreqCount]() -> std::string {
        for (auto [ip, slots] : hostFreqCount) {
            if (slots > 0) {
                return ip;
            }
        }

        // Unreachable (in this context)
        throw std::runtime_error("No next host with slots found!");
    };

    assert(newDecision->hosts.size() == oldDecision->hosts.size());

    // First we try to allocate to each message the same host they used to have
    for (int i = 0; i < oldDecision->hosts.size(); i++) {
        auto oldHost = oldDecision->hosts.at(i);

        if (hostFreqCount.contains(oldHost) && hostFreqCount.at(oldHost) > 0) {
            decision->addMessageInPosition(i,
                                           oldHost,
                                           oldDecision->messageIds.at(i),
                                           oldDecision->appIdxs.at(i),
                                           oldDecision->groupIdxs.at(i),
                                           oldDecision->mpiPorts.at(i));

            hostFreqCount.at(oldHost) -= 1;
        }
    }

    // Second we allocate the rest
    for (int i = 0; i < oldDecision->hosts.size(); i++) {
        if (decision->nFunctions <= i || decision->hosts.at(i).empty()) {

            auto nextHost = nextHostWithSlots();
            decision->addMessageInPosition(i,
                                           nextHost,
                                           oldDecision->messageIds.at(i),
                                           oldDecision->appIdxs.at(i),
                                           oldDecision->groupIdxs.at(i),
                                           -1);

            hostFreqCount.at(nextHost) -= 1;
        }
    }

    // Assert that we have preserved the new decision's host-message histogram
    // (use the pre-processor macro as we assert repeatedly in the loop, so we
    // want to avoid having an empty loop in non-debug mode)
#ifndef NDEBUG
    for (auto [host, freq] : hostFreqCount) {
        assert(freq == 0);
    }
#endif

    return decision;
}

// For the RoundRobin scheduler, a decision is better than another one if it spans
// less hosts. In case of a tie, we calculate the number of cross-VM links
// (i.e. better locality, or better packing)
bool RoundRobinScheduler::isFirstDecisionBetter(
  std::shared_ptr<SchedulingDecision> decisionA,
  std::shared_ptr<SchedulingDecision> decisionB)
{
    // The locality score is currently the number of cross-VM links. You may
    // calculate this number as follows:
    // - If the decision is single host, the number of cross-VM links is zero
    // - Otherwise, in a fully-connected graph, the number of cross-VM links
    //   is the sum of edges that cross a VM boundary
    auto getLocalityScore =
      [](std::shared_ptr<SchedulingDecision> decision) -> std::pair<int, int> {
        // First, calculate the host-message histogram (or frequency count)
        std::map<std::string, int> hostFreqCount;
        for (auto host : decision->hosts) {
            hostFreqCount[host] += 1;
        }

        // If scheduling is single host, return one host and 0 cross-host links
        if (hostFreqCount.size() == 1) {
            return std::make_pair(1, 0);
        }

        // Else, sum all the egressing edges for each element and divide by two
        int score = 0;
        for (auto [host, freq] : hostFreqCount) {

            int thisHostScore = 0;
            for (auto [innerHost, innerFreq] : hostFreqCount) {
                if (innerHost != host) {
                    thisHostScore += innerFreq;
                }
            }

            score += thisHostScore * freq;
        }

        score = int(score / 2);

        return std::make_pair(hostFreqCount.size(), score);
    };

    auto scoreA = getLocalityScore(decisionA);
    auto scoreB = getLocalityScore(decisionB);

    // The first decision is better if it has a LOWER host set size
    if (scoreA.first != scoreB.first) {
        return scoreA.first < scoreB.first;
    }

    // The first decision is better if it has a LOWER locality score
    return scoreA.second < scoreB.second;
}

std::vector<Host> RoundRobinScheduler::getSortedHosts(
  HostMap& hostMap,
  const InFlightReqs& inFlightReqs,
  std::shared_ptr<faabric::BatchExecuteRequest> req,
  const DecisionType& decisionType)
{
    std::vector<Host> sortedHosts;

    if (hostMap.empty()) {
        SPDLOG_ERROR("No hosts available for scheduling");
        return {};
    }
    
    for (const auto& [ip, host] : hostMap) {
        sortedHosts.push_back(host);
    }
    
    int nextHostIdx = (++nextHostCounter) % hostMap.size();
    std::rotate(sortedHosts.begin(), sortedHosts.begin() + nextHostIdx, sortedHosts.end());

    return sortedHosts;
}

// The RoundRobin's scheduler decision algorithm is very simple. It first sorts
// hosts (i.e. bins) in a specific order (depending on the scheduling type),
// and then starts filling bins from begining to end, until it runs out of
// messages to schedule
std::shared_ptr<SchedulingDecision> RoundRobinScheduler::makeSchedulingDecision(
  HostMap& hostMap,
  const InFlightReqs& inFlightReqs,
  std::shared_ptr<BatchExecuteRequest> req)
{
    auto decision = std::make_shared<SchedulingDecision>(req->appid(), 0);

    // Get the sorted list of hosts
    auto decisionType = getDecisionType(inFlightReqs, req);
    auto sortedHosts = getSortedHosts(hostMap, inFlightReqs, req, decisionType);

    // For an OpenMP request with the single host hint, we only consider
    // scheduling in one VM
    bool isOmp = req->messages_size() > 0 && req->messages(0).isomp();
    if (req->singlehosthint() && isOmp) {
        sortedHosts.erase(sortedHosts.begin() + 1, sortedHosts.end());
    }

    // Assign slots from the list (i.e. bin-pack)
    auto it = sortedHosts.begin();
    int numLeftToSchedule = req->messages_size();
    int msgIdx = 0;
    while (it < sortedHosts.end()) {
        // Calculate how many slots can we assign to this host (assign as many
        // as possible)
        int numOnThisHost =
          std::min<int>(numLeftToSchedule, numSlotsAvailable(*it));
        for (int i = 0; i < numOnThisHost; i++) {
            decision->addMessage(getIp(*it), req->messages(msgIdx));
            msgIdx++;
        }

        // Update the number of messages left to schedule
        numLeftToSchedule -= numOnThisHost;

        // If there are no more messages to schedule, we are done
        if (numLeftToSchedule == 0) {
            break;
        }

        // Otherwise, it means that we have exhausted this host, and need to
        // check in the next one
        it++;
    }

    // If we still have enough slots to schedule, we are out of slots
    if (numLeftToSchedule > 0) {
        return std::make_shared<SchedulingDecision>(NOT_ENOUGH_SLOTS_DECISION);
    }

    // In case of a DIST_CHANGE decision (i.e. migration), we want to make sure
    // that the new decision is better than the previous one
    if (decisionType == DecisionType::DIST_CHANGE) {
        auto oldDecision = inFlightReqs.at(req->appid()).second;
        if (isFirstDecisionBetter(decision, oldDecision)) {
            // If we are sending a better migration, make sure that we minimise
            // the number of migrations to be done
            return minimiseNumOfMigrations(decision, oldDecision);
        }

        return std::make_shared<SchedulingDecision>(DO_NOT_MIGRATE_DECISION);
    }

    return decision;
}
}
