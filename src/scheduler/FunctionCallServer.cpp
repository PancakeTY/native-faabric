#include <faabric/scheduler/FunctionCallServer.h>
#include <faabric/snapshot/SnapshotRegistry.h>
#include <faabric/state/State.h>
#include <faabric/util/config.h>
#include <faabric/util/func.h>
#include <faabric/util/logging.h>

#include <faabric/rpc/macros.h>

namespace faabric::scheduler {
FunctionCallServer::FunctionCallServer()
  : faabric::transport::MessageEndpointServer(DEFAULT_RPC_HOST,
                                              FUNCTION_CALL_PORT)
  , scheduler(getScheduler())
{}

void FunctionCallServer::doRecv(void* msgData, int size)
{
    throw std::runtime_error("doRecv for one message not implemented");
}

void FunctionCallServer::doRecv(const void* headerData,
                                int headerSize,
                                const void* bodyData,
                                int bodySize)
{
    int call = static_cast<int>(*static_cast<const char*>(headerData));
    switch (call) {
        case faabric::scheduler::FunctionCalls::MpiMessage:
            this->recvMpiMessage(bodyData, bodySize);
            break;
        case faabric::scheduler::FunctionCalls::Flush:
            this->recvFlush();
            break;
        case faabric::scheduler::FunctionCalls::ExecuteFunctions:
            this->recvExecuteFunctions(bodyData, bodySize);
            break;
        case faabric::scheduler::FunctionCalls::Unregister:
            this->recvUnregister(bodyData, bodySize);
            break;
        case faabric::scheduler::FunctionCalls::GetResources:
            this->recvGetResources(bodyData, bodySize);
            break;
        default:
            throw std::runtime_error(
              fmt::format("Unrecognized call header: {}", call));
    }
}

void FunctionCallServer::recvMpiMessage(const void* msgData, int size)
{
    // TODO - can we avoid this copy here?
    faabric::MPIMessage mpiMsg;

    // Deserialise message string
    if (!mpiMsg.ParseFromArray(msgData, size)) {
        throw std::runtime_error("Error deserialising message");
    }

    MpiWorldRegistry& registry = getMpiWorldRegistry();
    MpiWorld& world = registry.getWorld(mpiMsg.worldid());
    world.enqueueMessage(mpiMsg);
}

void FunctionCallServer::recvFlush()
{
    // Clear out any cached state
    faabric::state::getGlobalState().forceClearAll(false);

    // Clear the scheduler
    scheduler.flushLocally();

    // Reset the scheduler
    scheduler.reset();
}

void FunctionCallServer::recvExecuteFunctions(const void* msgData, int size)
{
    // TODO - avoiding having to copy the message here
    faabric::BatchExecuteRequest requestCopy;

    // Deserialise message string
    if (!requestCopy.ParseFromArray(msgData, size)) {
        throw std::runtime_error("Error deserialising message");
    }

    // This host has now been told to execute these functions no matter what
    scheduler.callFunctions(requestCopy, true);
}

void FunctionCallServer::recvUnregister(const void* msgData, int size)
{
    // TODO - avoiding having to copy the message here
    faabric::UnregisterRequest request;

    // Deserialise message string
    if (!request.ParseFromArray(msgData, size)) {
        throw std::runtime_error("Error deserialising message");
    }

    std::string funcStr =
      faabric::util::funcToString(request.function(), false);
    faabric::util::getLogger()->info(
      "Unregistering host {} for {}", request.host(), funcStr);

    // Remove the host from the warm set
    scheduler.removeRegisteredHost(request.host(), request.function());
}

void FunctionCallServer::recvGetResources(const void* data, int size)
{
    // Read the return address from the received data
    const std::string returnHost((const char*)data, size);

    // Prepare the response
    faabric::HostResources response = scheduler.getThisHostResources();

    // Open the endpoint socket, server always binds
    faabric::transport::SimpleMessageEndpoint endpoint(
      returnHost, FUNCTION_CALL_PORT + REPLY_PORT_OFFSET);
    endpoint.open(faabric::transport::getGlobalMessageContext(),
                  faabric::transport::SocketType::PUSH,
                  true);

    // Send the response body
    size_t responseSize = response.ByteSizeLong();
    // Deliberately use heap-allocation for zero-copy sending
    char* serialisedMsg = new char[responseSize];
    // Serialise using protobuf
    if (!response.SerializeToArray(serialisedMsg, responseSize)) {
        throw std::runtime_error("Error serialising message");
    }
    endpoint.send(serialisedMsg, responseSize);
}
}
